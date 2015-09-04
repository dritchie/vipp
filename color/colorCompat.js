
var _  = require('underscore');
var util = require('src/util');
var nnutil = require('src/neuralnet/utils');
var bounds = require('src/boundsTransforms');
var assert = require('assert');
var colorComp = require('colorcompatibility');
var colorSpaces = require('colorcompatibility/colorSpaces');
var colorUtils = require('color/utils');

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

// Possible layer transforms
var id = function(x) { return x; };
var neg_exp = function(x) { return Math.exp(-x); };
var sigmoid = function(x) { return 1 / (1 + Math.exp(-x)); };
var tanh = function(x) { return Math.tanh(x); };
var lecun_tanh = function(x) { return 1.7159 * Math.tanh(2/3 * x); };
var rectified_linear = function(x) { return Math.max(0, x); };

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

	// Architecture to use for neural nets
	var nn_arch = opt(opts.nn_arch, 8);

	// How much to fuzz the inputs of neural nets
	var nn_input_fuzz = opt(opts.nn_input_fuzz, 1e-8);

	// Which nn layer transform / activation function to use for hidden layers
	var nn_layer_transform = opt(opts.nn_layer_transform, lecun_tanh);

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
			return repeat(n, function(i) {
				var b = prm(undefined, undefined, gaussianERP.sample, unitNormalParams)
				var xform = _.isFunction(transform) ? transform : transform[i];
				return xform(b + layerSum(layerIndex, inputs));
			});
		}
	}

	// Single-layer perceptron
	var FUZZ = [0, nn_input_fuzz];
	var perceptron = function(inputs, nHidden, nOut, outTransform) {
		// Randomly 'fuzz' the inputs so that they aren't ever zero
		for (var i = 0; i < inputs.length; i++)
			inputs[i] = inputs[i] + gaussianERP.sample(FUZZ);
		var prevLayer = inputs;
		repeat(nHidden.length, function(i) {
			var n = nHidden[i];
			prevLayer = nnLayer(i, prevLayer, n, lecun_tanh);
		});
		return nnLayer(nHidden.length, prevLayer, nOut, outTransform);
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
	// Using 0.5 gives zero entropy derivative, so we randomize
	// (may eventually need to 'fuzz' all ERP param initial values for this very reason...)
	var _mixChoice_params = [gaussianERP.sample([0.5, 0.05])];
	var _mixChoice_bounds = [bounds.unitInterval];
	var _uniform = function(lo, hi, nninputs) {

		var params = erpGetParams(_uniform_params, _uniform_bounds, nninputs, neg_exp);
		var u = beta(params[0] + 0.01, params[1] + 0.01);

		// // Mixture of two betas
		// var flipParams = erpGetParams(_mixChoice_params, _mixChoice_bounds, nninputs, sigmoid, '[which]');
		// var which = flip(flipParams[0]) + 0;
		// var betaParams = erpGetParams(_uniform_params, _uniform_bounds, nninputs, neg_exp, '[' + which + ']');
		// var u = beta(betaParams[0] + 0.01, betaParams[1] + 0.01);

		// // Different way of doing a mixture of two betas
		// var params = erpGetParams(_mixChoice_params.concat(_uniform_params).concat(_uniform_params),
		// 	_mixChoice_bounds.concat(_uniform_bounds).concat(_uniform_bounds),
		// 	nninputs,
		// 	[sigmoid, neg_exp, neg_exp, neg_exp, neg_exp]);
		// var which = flip(params[0]) + 0;
		// var u = beta(params[2*which + 1] + 0.01, params[2*which + 2] + 0.01);

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

	var trainingData = colorUtils.loadTrainingData(3.7);

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

		// // Factor goes here
		// factorFunc(function() {
		// 	var score = colorComp.getRating(palette.map(cSpace.toRGB));
		// 	return gaussFactor(score, 5, 0.1);
		// });

		factorFunc(function() {
			var trainPalette = trainingData.palettes[randomIntegerERP.sample([trainingData.palettes.length])];
			var f = 0;
			for (var i = 0; i < 5; i++) {
				f = f + gaussFactor(palette[i][0], trainPalette[i][0], 0.1);
				f = f + gaussFactor(palette[i][1], trainPalette[i][1], 0.1);
				f = f + gaussFactor(palette[i][2], trainPalette[i][2], 0.1);
			}
			return f;
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
		erpGetParams = function(params, bounds, nninputs, outTransform, tag) {
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
			nninputs = nnutil.normalizeInputs(callsite, nninputs, inputSampleCache);
			if (tag !== undefined)
				callsite = callsite + tag;
			// Decide how many hidden layers, and how many units per layer
			var nHidden = [];
			for (var i = 0; i < nn_arch.length; i++) {
				var layer = nn_arch[i];
				if (layer.n !== undefined)
					nHidden.push(layer.n);
				else {
					var prevn = i === 0 ? nninputs.length : nHidden[i-1];
					var n = Math.ceil(Math.min(Math.max(layer.inputMult*prevn, layer.min), layer.max));
					nHidden.push(n);
				}
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
	nSteps: 400,
	convergeEps: 0.1,
	initLearnrate: 1,

	// optimizeMethod: 'Adam',
	// stepSize: 0.005,
	// adamEps: 1e-8
};

var target = makeProgram({family: 'target'});

// Single layer, fixed number of hidden units
var nnArch_fixed8 = [ { n: 8 } ];
var nnArch_fixed10 = [ { n: 10 } ];
var nnArch_fixed20 = [ { n: 20 } ];

// Two layers, fixed number of hidden units
var nnArch_fixed10_fixed5 = [ { n: 10 }, { n: 5 }];

// Three layers, fixed number of hidden units
var nnArch_fixed12_fixed6_fixed3 = [ { n: 12 }, { n: 6 }, { n: 3 }];

// Four layers!
var nnArch_fixed3x4 = [{n:3},{n:3},{n:3},{n:3}];

// Single layer, number of hidden units = number of inputs
var nnArch_relative_oneLayer = [ {
	inputMult: 1,
	min: 1,
	max: 100
} ];

// Two layers, number of hidden units = number of inputs
var nnArch_relative_twoLayer = [
	{
		inputMult: 1,
		min: 1,
		max: 100
	},
	{
		inputMult: 1,
		min: 1,
		max: 100
	}
];

var nnArch = nnArch_relative_oneLayer;
var fuzzAmt = 1e-1;
var maxNorm = 3;
var entRegWeight = 5;

// ----------------------------------------------------------------------------

// EARLY EXPERIMENTS

var params = {};

// Mean field
params['meanField'] = { guideParams: { family: 'meanField' } };

// Neural, no other modifications
params['neural_baseline'] = { guideParams: {
	family: 'neural',
	nn_arch: nnArch
}};

// Neural + significant fuzz on NN input layer
params['neural_fuzz_' + fuzzAmt] = { guideParams : {
	family: 'neural',
	nn_arch: nnArch,
	nn_input_fuzz: fuzzAmt
}};

// Neural + max-norm regularization
params['neural_maxNorm_' + maxNorm] = { guideParams : {
	family: 'neural',
	nn_arch: nnArch,
	max_norm_regularization: true,
	max_norm: maxNorm
}};

// Neural + dropout
params['dropout'] = {
	guideParams: {
		family: 'neural',
		nn_arch: nnArch,
		dropout_phase: 'train'
	},
	testGuideParams: {
		family: 'neural',
		nn_arch: nnArch,
		dropout_phase: 'test'
	},
	variationalParams: { allowZeroDerivatives: true }
};

// Neural + entropy regularization
params['entropyReg_' + entRegWeight] = {
	guideParams: { family: 'neural', nn_arch: nnArch },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};


// ----------------------------------------------------------------------------

// EXPERIMENTS W/ ENTROPY REGULARIZATION + DIFFERENT ARCHITECTURES

// Mean field
params['meanField_entReg'] = {
	guideParams: { family: 'meanField' },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};

// One layer
params['oneLayer'] = {
	guideParams: { family: 'neural', nn_arch: nnArch_relative_oneLayer },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};

// One layer, rectified linear units
params['oneLayer_rectLin'] = {
	guideParams: { family: 'neural', nn_arch: nnArch_relative_oneLayer, nn_layer_transform: rectified_linear },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};

// One layer, twice as many hidden nodes as inputs
params['oneLayer_inputMult1.5'] = {
	guideParams: { family: 'neural', nn_arch: [{inputMult: 1.5, min: 1, max: 100}] },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};

// One layer, fixed number of nodes
params['oneLayer_n10'] = {
	guideParams: { family: 'neural', nn_arch: nnArch_fixed10 },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};
params['oneLayer_n20'] = {
	guideParams: { family: 'neural', nn_arch: nnArch_fixed20 },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};

// Two layer
params['twoLayer'] = {
	guideParams: { family: 'neural', nn_arch: nnArch_relative_twoLayer },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};
params['twoLayer_n10n5'] = {
	guideParams: { family: 'neural', nn_arch: nnArch_fixed10_fixed5 },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};

// Three layer
params['threeLayer_n12n6n3'] = {
	guideParams: { family: 'neural', nn_arch: nnArch_fixed12_fixed6_fixed3 },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};

// Four layer
params['fourLayer_n3x4'] = {
	guideParams: { family: 'neural', nn_arch: nnArch_fixed3x4 },
	variationalParams: { entropyRegularizeWeight: entRegWeight }
};

// Six layer
params['sixLayer'] = {
	guideParams: { family: 'neural', nn_arch: repeat(6, function() { return {n:3}; }) },
	// variationalParams: { entropyRegularizeWeight: entRegWeight }
};

// ----------------------------------------------------------------------------


var nSamps = 20;

var runTest = function(name, allowZeroDerivatives, outputName) {
	if (outputName === undefined) outputName = name;
	var p = params[name];
	var guide = makeProgram(p.guideParams);
	var testGuide = p.testGuideParams === undefined ? guide : makeProgram(p.testGuideParams);
	var variationalParams = p.variationalParams === undefined ? defaultParams :
								_.extend(_.clone(defaultParams), p.variationalParams);
	if (allowZeroDerivatives)
		variationalParams = _.extend(_.clone(variationalParams), {allowZeroDerivatives: true});
	var result = variational.infer(target, guide, undefined, variationalParams);
	var palettes = repeat(nSamps, function() {
		return util.runWithAddress(testGuide, '', [result.params]);
	});
	var otherOpts = {
		finalELBO: result.elbo,
		stepsTaken: result.stepsTaken,
		timeTaken: result.timeTaken,
		variationalParams: variationalParams,
		guideParams: p.guideParams
	};
	if (p.testGuideParams !== undefined) {
		otherOpts.testGuideParams = p.testGuideParams;
	}
	colorUtils.saveStatsAndSamples(outputName, palettes, otherOpts);
	variational.saveParams(result.params, 'color/results/' + outputName + '/params.txt');
};

var runNaiveForward = function() {
	var palettes = repeat(nSamps, function() {
		return util.runWithAddress(target, '');
	});
	colorUtils.saveStatsAndSamples('naiveForward', palettes);
};

// ----------------------------------------------------------------------------

// runNaiveForward();
// runTest('meanField', true, 'derp');
// runTest('meanField_entReg', true, 'meanField_entReg5_3');
// runTest('neural_baseline', true, 'oneLayer');
// runTest('oneLayer', true, 'oneLayer_entReg5_mixtureSameNet_3');
// runTest('oneLayer_rectLin', true, 'oneLayer_entReg5_rectLin_3');
// runTest('oneLayer_n10', true, 'oneLayer_entReg5_n10_3');
// runTest('twoLayer_n10n5', true, 'twoLayer_entReg5_n10n5_3');
// runTest('threeLayer_n12n6n3', true, 'threeLayer_entReg5_n12n6n3_3');
// runTest('fourLayer_n3x4', true, 'fourLayer_n3x4_3');
runTest('sixLayer', true);







