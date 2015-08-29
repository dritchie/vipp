
var colorComp = require('colorcompatibility');
var util = require('src/util');
var nnutil = require('src/neuralnet/utils');
var bounds = require('src/boundsTransforms');


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

var makeProgram = function(family) {

	// Global parameters we might want to fiddle with

	// How many hidden layer nodes to use for the ERP parameter neural nets
	var ERP_NHIDDEN = 4;

	// How many neural net input samples should we collect at each callsite
	//    to determine how to normalize the input?
	var NUM_NORMALIZE_SAMPLES = 100;
	var inputSampleCache = nnutil.makeInputSampleCache(NUM_NORMALIZE_SAMPLES);

	// ----------------------------------------------------------------------------

	// Parameter function
	var prm;

	// ----------------------------------------------------------------------------

	// Neural network stuff

	var unitNormalParams = [0, 1];
	var weightedSum = function(nums, i) {
		if (i === undefined) i = 0;
		// Originally, I was initializing params to zero, but that doesn't work, because
		//    then all the derivatives are zero.
		// return i === nums.length ? 0 : prm(0) * nums[i] + weightedSum(nums, i + 1);
		return i === nums.length ? 0 : prm(undefined, undefined, gaussianERP.sample, unitNormalParams) * nums[i] + weightedSum(nums, i + 1);
	}

	// Possible layer transforms
	var id = function(x) { return x; };
	var neg_exp = function(x) { return Math.exp(-x); };
	var sigmoid = function(x) { return 1 / (1 + Math.exp(-x)); };
	var tanh = function(x) { return Math.tanh(x); };
	var lecun_tanh = function(x) { return 1.7159 * Math.tanh(2/3 * x); };

	var nnLayer = function(inputs, n, transform) {
		return repeat(n, function() {
			return transform(weightedSum(inputs));
		});
	}

	// Single-layer perceptron
	var FUZZ = [0, 1e-8];
	var perceptron = function(inputs, nHidden, nOut, outTransform) {
		// Randomly 'fuzz' the inputs so that they aren't ever zero
		for (var i = 0; i < inputs.length; i++)
			inputs[i] = inputs[i] + gaussianERP.sample(FUZZ);
		var hiddenLayer = nnLayer(inputs, nHidden, lecun_tanh);
		return nnLayer(hiddenLayer, nOut, outTransform);
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
		var u = beta(params[0], params[1]);
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
		prm = function(v, t, s, h) { return param(params, v, t, s, h); };

		// TODO: Try sampling in different color spaces?

		var r1 = _uniform(0, 1);
		var g1 = _uniform(0, 1, nnInput(r1));
		var b1 = _uniform(0, 1, nnInput(r1, g1));

		var r2 = _uniform(0, 1, nnInput(r1, g1, b1));
		var g2 = _uniform(0, 1, nnInput(r1, g1, b1, r2));
		var b2 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2));

		var r3 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2, b2));
		var g3 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2, b2, r3));
		var b3 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2, b2, r3, g3));

		var r4 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3));
		var g4 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4));
		var b4 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4));

		var r5 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4));
		var g5 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5));
		var b5 = _uniform(0, 1, nnInput(r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5, g5));

		var palette = [
			[r1, g1, b1],
			[r2, g2, b2],
			[r3, g3, b3],
			[r4, g4, b4],
			[r5, g5, b5]
		];

		// Factor goes here
		factorFunc(function() {
			var score = colorComp.getRating(palette);
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
			return util.runWithAddress(perceptron, callsite, [nninputs, ERP_NHIDDEN, params.length, outTransform]);
		}; 
	}

	return generate;
};


// ----------------------------------------------------------------------------

var name = 'test';

// Mean field variational test
var target = makeProgram('target');
// var guide = makeProgram('meanField');
var guide = makeProgram('neural');
var result = variational.infer(target, guide, undefined, {
	verbosity: 3,
	nSamples: 100,
	nSteps: 200,
	convergeEps: 0.1,
	initLearnrate: 0.5
});
variational.saveParams(result.params, 'color/results/'+name+'.params');
// var result = { params: variational.loadParams('color/results/'+name+'.params') };
for (var i = 0; i < 10; i++) {
	var palette = util.runWithAddress(guide, '', [result.params]);
	// var palette = util.runWithAddress(target, '');
	console.log(colorComp.getRating(palette));
	require('color/utils').drawPalette(palette, 'color/results/' + name + '_' + i + '.png');
}



