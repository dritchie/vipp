
var _  = require('underscore');
var colorComp = require('colorcompatibility');
var colorSpaces = require('colorcompatibility/colorSpaces');
var util = require('src/util');
var nnutil = require('src/neuralnet/utils');
var bounds = require('src/boundsTransforms');

var map = function(fn, ar) {
  return ar.length === 0 ? [] : [fn(ar[0])].concat(map(fn, ar.slice(1)));
};

var mapIndexed = function(f, l) {
  var fn = function(ls, i, _acc) {
    return ls.length === 0 ?
        _acc :
        fn(ls.slice(1), i + 1, _acc.concat([f(i, ls[0])]))
  };
  return fn(l, 0, []);
}

var repeat = function(n, fn) {
  return n === 0 ? [] : repeat(n - 1, fn).concat([fn(n - 1)]);
};

// ----------------------------------------------------------------------------

// Which color space should we sample in?

var cSpace = colorSpaces.RGB;
// var cSpace = colorSpaces.HSV;
// var cSpace = colorSpaces.LAB;

var channelBounds;
if (cSpace === colorSpaces.RGB) {
	channelBounds = [
		{lo: 0, hi: 1},
		{lo: 0, hi: 1},
		{lo: 0, hi: 1},
	];
} else if (cSpace === colorSpaces.HSV) {
	channelBounds = [
		{lo: 0, hi: 360},
		{lo: 0, hi: 1},
		{lo: 0, hi: 1},
	];
} else if (cSpace === colorSpaces.LAB) {
	channelBounds = [
		{lo: 0, hi: 100},
		{lo: -100, hi: 100},
		{lo: -100, hi: 100},
	];
}

// ----------------------------------------------------------------------------

var makeProgram = function(opts) {

	if (opts === undefined) opts = {};
	var opt = function(val, defaultval) {
		if (val === undefined && opts === undefined)
			throw 'Option undefined and has no default value!';
		return val === undefined ? defaultval : val;
	}

	// Global parameters we might want to fiddle with

	var family = opt(opts.family);

	// How many hidden layer nodes to use for the ERP parameter neural nets
	var erp_n_hidden = opt(opts.erp_n_hidden, 8);

	// How much to fuzz the inputs of neural nets
	var nn_input_fuzz = opt(opts.nn_input_fuzz, 1e-8);

	// How many neural net input samples should we collect at each callsite
	//    to determine how to normalize the input?
	var num_normalize_samples = opt(opts.num_normalize_samples, 100);
	var inputSampleCache = nnutil.makeInputSampleCache(num_normalize_samples);

	// max-norm regularization
	var max_norm_regularization = opt(opts.max_norm_regularization, false);
	var max_norm = opt(opts.max_norm, 4);

	// dropout
	var dropout_phase = opt(opts.dropout_phase, undefined);
	var dropout_input_p = opt(opts.dropout_input_p, 0.8);
	var dropout_hidden_p = opt(opts.dropout_hidden_p, 0.5);

	// ----------------------------------------------------------------------------

	var globalStore = {};

	// ----------------------------------------------------------------------------

	// Parameter function
	var prm = function(v, t, s, h) { return param(globalStore.params, v, t, s, h); };

	// ----------------------------------------------------------------------------

	// Neural network stuff

	// Dot product of weights / inputs at a NN node
	var unitNormalParams = [0, 1];
	var layerSum;
	if (max_norm_regularization) {
		layerSum = function(layerIndex, nums) {
			// Retrieve the weights
			var paramNames = [];
			var norm = 0;
			var wmult = (dropout_phase === 'test') ? (layerIndex === 0 ? dropout_input_p : dropout_hidden_p) : 1;
			var weights = map(function(x) {
				var p = wmult * prm(undefined, undefined, gaussianERP.sample, unitNormalParams, paramNames);
				norm += ad_primal(p) * ad_primal(p);
				return p;
			}, nums);
			// Do max-norm regularization
			norm = Math.sqrt(norm);
			if (norm > max_norm) {
				var factor = norm / max_norm
				for (var i = 0; i < weights.length; i++) {
					weights[i] = weights[i] / factor;
					globalStore.params[paramNames[i]] = weights[i];
				}
			}
			// Finally, compute dot product
			var s = 0;
			for (var i = 0; i < nums.length; i++)
				s = s + nums[i]*weights[i];
			return s;
		}
	} else {
		layerSum = function(layerIndex, nums) {
			// Retrieve the weights
			var wmult = (dropout_phase === 'test') ? (layerIndex === 0 ? dropout_input_p : dropout_hidden_p) : 1;
			var weights = map(function(x) {
				return wmult * prm(undefined, undefined, gaussianERP.sample, unitNormalParams);
			}, nums);
			// Finally, compute dot product
			var s = 0;
			for (var i = 0; i < nums.length; i++)
				s = s + nums[i]*weights[i];
			return s;
		}
	}

	// Possible layer transforms
	var id = function(x) { return x; };
	var neg_exp = function(x) { return Math.exp(-x); };
	var sigmoid = function(x) { return 1 / (1 + Math.exp(-x)); };
	var tanh = function(x) { return Math.tanh(x); };
	var lecun_tanh = function(x) { return 1.7159 * Math.tanh(2/3 * x); };

	// Computing one layer of a neural net
	var nnLayer;
	if (dropout_phase === 'train') {
		nnLayer = function(layerIndex, inputs, n, transform) {
			// Randomly mask the inputs
			var p = layerIndex === 0 ? dropout_input_p : dropout_hidden_p;
			for (var i = 0; i < inputs.length; i++)
				inputs[i] = (bernoulliERP.sample([p]) + 0) * inputs[i];
			return repeat(n, function() {
				var b = prm(undefined, undefined, gaussianERP.sample, unitNormalParams)
				return transform(b + layerSum(layerIndex, inputs));
			});
		}
	} else {
		nnLayer = function(layerIndex, inputs, n, transform) {
			return repeat(n, function() {
				var b = prm(undefined, undefined, gaussianERP.sample, unitNormalParams)
				return transform(b + layerSum(layerIndex, inputs));
			});
		}
	}

	// Single-layer perceptron
	var FUZZ = [0, nn_input_fuzz];
	var perceptron = function(inputs, nHidden, nOut, outTransform) {
		// Randomly 'fuzz' the inputs so that they aren't ever zero
		for (var i = 0; i < inputs.length; i++)
			inputs[i] = inputs[i] + gaussianERP.sample(FUZZ);
		// if (nHidden === 'inputSize') nHidden = Math.max(inputs.length, 4);
		var hiddenLayer = nnLayer(0, inputs, nHidden, lecun_tanh);
		return nnLayer(1, hiddenLayer, nOut, outTransform);
	}

	var nnInput;
	if (family === 'neural')
		nnInput = function() {
			// arguments[0] is the address
			return Array.prototype.slice.call(arguments, 1);
		};
	else nnInput = function() {};

	// ----------------------------------------------------------------------------

	// ERP stuff

	// Get ERP parameters, depending on which program family we're using
	var erpGetParams;
	if (family === 'target')
		erpGetParams = function(params) { return params; };
	else if (family === 'meanField')
		erpGetParams = function(params, bounds) {
			return mapIndexed(function(i, p) { return prm(params[i], bounds[i]); }, params);
		};
	else if (family === 'neural') {
		// We start out with a version that just collects stats about the NN inputs
		// Later, we'll swap this out with a version that actually uses the NNs.
		erpGetParams = function(params, bounds, nninputs, outTransform) {
			// For now, neural nets share parameters per callsite
			var address = arguments[0];
			if (nninputs === undefined) {
				// console.log('address: ' + address);
				// throw 'ERP param neural net has no inputs!';

				return params;
			}
			var addrParts = address.split('_');
			var callsite = '_' + addrParts[addrParts.length - 2];
			nnutil.collectSample(callsite, nninputs, inputSampleCache);
			return params;
		}; 
	}

	var _uniform_params = [1, 1];
	var _uniform_bounds = [bounds.nonNegative, bounds.nonNegative];
	var _uniform = function(lo, hi, nninputs) {
		var params = erpGetParams(_uniform_params, _uniform_bounds, nninputs, neg_exp);
		var u = beta(params[0] + 0.01, params[1] + 0.01);
		return (1.0-u)*lo + u*hi;
	}

	// ----------------------------------------------------------------------------

	var factorFunc;
	if (family === 'target') {
		factorFunc = function(fn) { factor(fn()); };
	} else {
		factorFunc = function(fn) {};
	}

	var gaussFactor = function(x, mu, sigma) {
		return gaussianERP.score([mu, sigma], x);
	}

	// ----------------------------------------------------------------------------

	var generate = function(params) {
		globalStore.params = params;

		var r1 = _uniform(channelBounds[0].lo, channelBounds[0].hi);
		var g1 = _uniform(channelBounds[1].lo, channelBounds[1].hi, nnInput(r1));
		var b1 = _uniform(channelBounds[2].lo, channelBounds[2].hi, nnInput(r1, g1));

		var r2 = _uniform(channelBounds[0].lo, channelBounds[0].hi, nnInput(r1, g1, b1));
		var g2 = _uniform(channelBounds[1].lo, channelBounds[1].hi, nnInput(r1, g1, b1, r2));
		var b2 = _uniform(channelBounds[2].lo, channelBounds[2].hi, nnInput(r1, g1, b1, r2, g2));

		var r3 = _uniform(channelBounds[0].lo, channelBounds[0].hi, nnInput(r1, g1, b1, r2, g2, b2));
		var g3 = _uniform(channelBounds[1].lo, channelBounds[1].hi, nnInput(r1, g1, b1, r2, g2, b2, r3));
		var b3 = _uniform(channelBounds[2].lo, channelBounds[2].hi, nnInput(r1, g1, b1, r2, g2, b2, r3, g3));

		var r4 = _uniform(channelBounds[0].lo, channelBounds[0].hi, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3));
		var g4 = _uniform(channelBounds[1].lo, channelBounds[1].hi, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4));
		var b4 = _uniform(channelBounds[2].lo, channelBounds[2].hi, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4));

		var r5 = _uniform(channelBounds[0].lo, channelBounds[0].hi, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4));
		var g5 = _uniform(channelBounds[1].lo, channelBounds[1].hi, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5));
		var b5 = _uniform(channelBounds[2].lo, channelBounds[2].hi, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5, g5));

		var palette = [
			[r1, g1, b1],
			[r2, g2, b2],
			[r3, g3, b3],
			[r4, g4, b4],
			[r5, g5, b5]
		];

		// Factor goes here
		factorFunc(function() {
			var score = colorComp.getRating(palette.map(cSpace.toRGB));
			return gaussFactor(score, 5, 0.1);
		});

		return palette;
	};

	// ----------------------------------------------------------------------------

	if (family === 'neural') {
		// Run 'generate' repeatedly until we've collected enough NN input samples
		//    for all callsites to be able to normalize the inputs reasonably well.
		do {
			generate(variational.makeParams());
		} while (!inputSampleCache.hasEnoughSamples())

		// Now we can swap out the params function with the one that actually uses neural nets
		erpGetParams = function(params, bounds, nninputs, outTransform) {
			// For now, neural nets share parameters per callsite
			var address = arguments[0];
			if (nninputs === undefined) {
				// console.log('address: ' + address);
				// throw 'ERP param neural net has no inputs!';

				// Just do what mean-field does
				return mapIndexed(function(i, p) { return prm(params[i], bounds[i]); }, params);
			}
			var addrParts = address.split('_');
			var callsite = '_' + addrParts[addrParts.length - 2];
			nnutil.normalizeInputs(callsite, nninputs, inputSampleCache);
			// Decide how many hidden nodes the network should have
			var nHidden;
			if (erp_n_hidden.n !== undefined)
				nHidden = erp_n_hidden.n;
			else {
				nHidden = Math.ceil(Math.min(Math.max(erp_n_hidden.inputMult * nninputs.length, erp_n_hidden.min), erp_n_hidden.max));
			}
			return util.runWithAddress(perceptron, callsite, [nninputs, nHidden, params.length, outTransform]);
		}; 
	}

	return generate;
};

// ----------------------------------------------------------------------------

var defaultParams = {
	verbosity: 3,
	nSamples: 100,
	// nSteps: 200,
	nSteps: 400,
	convergeEps: 0.1,
	initLearnrate: 1,

	// DON'T LEAVE THIS IN!
	// allowZeroDerivatives: true
};

var target = makeProgram({family: 'target'});

// var nHidden =  { n: 8 };
var nHidden = {
	inputMult: 1,
	// inputMult: 1.5,
	min: 1,
	max: 100
};

var fuzzAmt = 1e-1;
var maxNorm = 3;
var entRegWeight = 5;
// var entRegWeight = 10;


var params = {};

params['meanField'] = { guideParams: { family: 'meanField' } };

params['neural_baseline'] = { guideParams: {
	family: 'neural',
	erp_n_hidden: nHidden
}};

params['neural_fuzz_' + fuzzAmt] = { guideParams : {
	family: 'neural',
	erp_n_hidden: nHidden,
	nn_input_fuzz: fuzzAmt
}};

params['neural_maxNorm_' + maxNorm] = { guideParams : {
	family: 'neural',
	erp_n_hidden: nHidden,
	max_norm_regularization: true,
	max_norm: maxNorm
}};

params['dropout'] = {
	guideParams: {
		family: 'neural',
		erp_n_hidden: nHidden,
		dropout_phase: 'train'
	},
	testGuideParams: {
		family: 'neural',
		erp_n_hidden: nHidden,
		dropout_phase: 'test'
	},
	variationalParams: { allowZeroDerivatives: true }
};

params['entropyReg_' + entRegWeight] = {
	guideParams: { family: 'neural', erp_n_hidden: nHidden },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};


var fs = require('fs');
var cp = require('child_process');

var runTest = function(name) {
	var p = params[name];
	var guide = makeProgram(p.guideParams);
	var testGuide = p.testGuideParams === undefined ? guide : makeProgram(p.testGuideParams);
	var variationalParams = p.variationalParams === undefined ? defaultParams :
								_.extend(_.clone(defaultParams), p.variationalParams);
	var result = variational.infer(target, guide, undefined, variationalParams);
	var dirname = 'color/results/' + name;
	if (fs.existsSync(dirname))
		cp.execSync('rm -rf ' + dirname);
	fs.mkdirSync(dirname);
	variational.saveParams(result.params, dirname + '/params.txt');
	for (var i = 0; i < 10; i++) {
		var palette = util.runWithAddress(testGuide, '', [result.params]);
		var rating = colorComp.getRating(palette.map(cSpace.toRGB));
		require('color/utils').drawPalette(palette, dirname + '/' + rating + '.png');
	}
};

var runNaiveForward = function() {
	var dirname = 'color/results/naiveForward';
	if (fs.existsSync(dirname))
		cp.execSync('rm -rf ' + dirname);
	fs.mkdirSync(dirname);
	for (var i = 0; i < 10; i++) {
		var palette = util.runWithAddress(target, '', []);
		var rating = colorComp.getRating(palette);
		require('color/utils').drawPalette(palette, dirname + '/' + rating + '.png');
	}
};

// runTest('meanField');
// runTest('neural_baseline');
// runTest('neural_fuzz_' + fuzzAmt);
// runTest('neural_maxNorm_' + maxNorm);
// runTest('dropout');
runTest('entropyReg_' + entRegWeight);

// runNaiveForward();







