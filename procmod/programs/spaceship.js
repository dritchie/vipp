var THREE = require('three');
var Geo = require('procmod/lib/geometry');
var SpaceshipUtil = require('procmod/programs/spaceshipUtil');
var bounds = require('src/boundsTransforms');
var util = require('src/util');
var nnutil = require('src/neuralnet/utils');


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

// Only works on one argument
var memoize = function(fn) {
	var cache = {};
	return function(x) {
		var y = cache[x];
		if (y === undefined) {
			y = fn(x);
			cache[x] = y;
		}
		return y;
	};
}

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

	var globalStore = {};

	// ----------------------------------------------------------------------------

	var addGeometry = function(geo) {
		globalStore.geometry.push(geo);
	}

	var addVolume = function(v) {
		globalStore.volume += v;
	}


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
			var res = [globalStore.volume];
			if (arguments.length > 1)  // arguments[0] is the address
				res = res.concat(Array.prototype.slice.call(arguments, 1));
			return res;
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
				console.log('address: ' + address);
				throw 'ERP param neural net has no inputs!';
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

	var _flip_bounds = [bounds.unitInterval];
	var _flip = function(p, nninputs) {
		p = Math.min(Math.max(0.01, p), 0.99);  // Can't be exactly 0 or exactly 1
		var params = erpGetParams([p], _flip_bounds, nninputs, sigmoid);
		return flip(params[0]);
	}

	var _discrete_bounds = memoize(function(n) { return repeat(n, function(x) { return bounds.unitInterval; }); });
	var _discrete = function(probs, nninputs) {
		var params = erpGetParams(probs, _discrete_bounds(probs.length), nninputs, sigmoid);
		return discrete(params);
	}

	// ----------------------------------------------------------------------------

	// Subroutines for random spaceship geometry generation.

	var wi = function(i, w) { return Math.exp(-w*i); }

	var addBoxBodySeg = function(rearz, prev) {
		// Must be bigger than the previous segment, if the previous
		//   segment was not a box (i.e. was a cylinder-type thing)
		// var xl = _uniform(1, 3);
		var xl = _uniform(1, 3, nnInput(rearz));
		var xlen = (prev.type === BodyType.Box) ? xl : Math.max(xl, prev.xlen);
		// var yl = _uniform(.5, 1) * xl;
		var yl = _uniform(.5, 1, nnInput(rearz, xlen)) * xl;
		var ylen = (prev.type === BodyType.Box) ? yl : Math.max(yl, prev.ylen);
		// var zlen = _uniform(2, 5);
		var zlen = _uniform(2, 5, nnInput(rearz, xlen, ylen));
		var geo = Geo.Shapes.Box(0, 0, rearz + 0.5*zlen, xlen, ylen, zlen);
		addGeometry(geo);
		addVolume(xlen*ylen*zlen);
		return { xlen: xlen, ylen: ylen, zlen: zlen, type: BodyType.Box };
	}

	var addCylinderBodySeg = function(rearz, prev, isnose) {
		// Must be smaller than previous segment, if that was a box
		var limitrad = 0.5*Math.min(prev.xlen, prev.ylen);
		var minrad = (prev.type === BodyType.Box) ? 0.4*limitrad : 0.3;
		var maxrad = (prev.type === BodyType.Box) ? limitrad : 1.25;
		// var radius = _uniform(minrad, maxrad);
		var radius = _uniform(minrad, maxrad, nnInput(rearz));
		var xlen = radius*2;
		var ylen = radius*2;
		// var zlen = isnose ? _uniform(1, 3) : _uniform(2, 5);
		var zlen = isnose ? _uniform(1, 3, nnInput(radius, rearz)) : _uniform(2, 5, nnInput(radius, rearz));
		var geo = isnose ? SpaceshipUtil.BodyCylinder(rearz, zlen, radius,
													 radius*_uniform(.25, .75, nnInput(radius, rearz, zlen)))
						 : SpaceshipUtil.BodyCylinder(rearz, zlen, radius);
		addGeometry(geo);
		addVolume(radius*radius*Math.PI*zlen);
		return { xlen: xlen, ylen: ylen, zlen: zlen, type: BodyType.Cylinder };
	}

	var addClusterBodySeg = function(rearz, prev, isnose) {
		// Must be smaller than previous segment, if that was a box
		var limitrad = 0.25*Math.min(prev.xlen, prev.ylen);
		var minrad = (prev.type === BodyType.Box) ? 0.4*limitrad : 0.5*0.3;
		var maxrad = (prev.type === BodyType.Box) ? limitrad : 0.5*1.25;
		// var radius = _uniform(minrad, maxrad);
		var radius = _uniform(minrad, maxrad, nnInput(rearz));
		var xlen = radius*4;
		var ylen = radius*4;
		// var zlen = _uniform(2, 5);
		var zlen = _uniform(2, 5, nnInput(radius, rearz));
		var geo = SpaceshipUtil.BodyCluster(rearz, zlen, radius);
		addGeometry(geo);
		addVolume(radius*radius*Math.PI*zlen*4);
		return { xlen: xlen, ylen: ylen, zlen: zlen, type: BodyType.Cluster };
	}

	var BodyType = { Box: 0, Cylinder: 1, Cluster: 2, N: 3 }
	var addBodySeg = function(rearz, prev) {	
		// var type = _discrete([.33, .33, .33]);
		var type = _discrete([.33, .33, .33], nnInput(rearz));
		if (type == BodyType.Box)
			return addBoxBodySeg(rearz, prev)
		else if (type == BodyType.Cylinder)
			return addCylinderBodySeg(rearz, prev);
		else if (type == BodyType.Cluster)
			return addClusterBodySeg(rearz, prev);
		else throw('unsupported body type ' + type);
	}

	var addBoxWingSeg = function(xbase, zlo, zhi) {
		// var zbase = _uniform(zlo, zhi);
		// var xlen = _uniform(0.25, 2.0);
		// var ylen = _uniform(0.25, 1.25);
		// var zlen = _uniform(0.5, 4.0);
		var zbase = _uniform(zlo, zhi, nnInput(xbase));
		var xlen = _uniform(0.25, 2.0, nnInput(xbase, zbase));
		var ylen = _uniform(0.25, 1.25, nnInput(xbase, zbase, xlen));
		var zlen = _uniform(0.5, 4.0, nnInput(xbase, zbase, xlen, ylen));
		var geo = SpaceshipUtil.WingBoxes(xbase, zbase, xlen, ylen, zlen);
		addGeometry(geo);
		addVolume(xlen*ylen*zlen*2);
		// if (_flip(0.5))
		if (_flip(0.5, nnInput(xbase, zbase, xlen, ylen, zlen)))
			addWingGuns(xbase, zbase, xlen, ylen, zlen);
		return { xlen: xlen, ylen: ylen, zlen: zlen, zbase: zbase };
	}

	var addWingGuns = function(xbase, zbase, xlen, ylen, zlen) {
		// var gunlen = _uniform(1, 1.2)*zlen;
		var gunlen = _uniform(1, 1.2, nnInput(xbase, zbase, xlen, ylen, zlen))*zlen;
		var gunxbase = xbase + 0.5*xlen;
		var gunybase = 0.5*ylen;
		var geo = SpaceshipUtil.WingGuns(gunxbase, gunybase, zbase, gunlen);
		addGeometry(geo);
		// Let's just say that guns don't count for overall volume...
	};

	var addCylinderWingSeg = function(xbase, zlo, zhi) {
		// var zbase = _uniform(zlo, zhi);
		// var radius = _uniform(.15, .7);
		// var zlen = _uniform(1, 5);
		var zbase = _uniform(zlo, zhi, nnInput(xbase));
		var radius = _uniform(.15, .7, nnInput(xbase, zbase));
		var zlen = _uniform(1, 5, nnInput(xbase, zbase, radius));
		var xlen = 2*radius;
		var ylen = 2*radius;
		var geo = SpaceshipUtil.WingCylinders(xbase, zbase, zlen, radius);
		addGeometry(geo);
		addVolume(radius*radius*Math.PI*zlen*2);
		return { xlen: xlen, ylen: ylen, zlen: zlen, zbase: zbase };
	}

	var WingType = { Box: 0, Cylinder: 1, N: 2 }
	var addWingSeg = function(xbase, zlo, zhi) {
		// var type = _flip(0.5) + 0;
		var type = _flip(0.5, nnInput(xbase)) + 0;
		if (type == WingType.Box)
			return addBoxWingSeg(xbase, zlo, zhi);
		else if (type == WingType.Cylinder)
			return addCylinderWingSeg(xbase, zlo, zhi);
		else throw('unsupported wing type ' + type);
	}

	var addWings = function(i, xbase, zlo, zhi) {
		var rets = addWingSeg(xbase, zlo, zhi);
		var xlen = rets.xlen;
		var ylen = rets.ylen;	
		var zlen = rets.zlen;
		var zbase = rets.zbase;
		// if (_flip(wi(i, 0.6)))
		if (_flip(wi(i, 0.6), nnInput(xbase)))
			addWings(i+1, xbase+xlen, zbase-0.5*zlen, zbase+0.5*zlen);
	}

	var addFin = function(i, ybase, zlo, zhi, xmax) {
		// var xlen = _uniform(0.5, 1.0) * xmax;
		// var ylen = _uniform(0.1, 0.5);
		// var zlen = _uniform(0.5, 1.0) * (zhi - zlo);
		var xlen = _uniform(0.5, 1.0, nnInput(ybase, xmax)) * xmax;
		var ylen = _uniform(0.1, 0.5, nnInput(ybase, xmax, xlen));
		var zlen = _uniform(0.5, 1.0, nnInput(ybase, xmax, ylen)) * (zhi - zlo);
		var zbase = 0.5*(zlo + zhi);
		var geo = Geo.Shapes.Box(0, ybase + 0.5*ylen, zbase, xlen, ylen, zlen);
		addGeometry(geo);
		addVolume(xlen*ylen*zlen);
		// if (_flip(wi(i, 0.2)))
		if (_flip(wi(i, 0.2), nnInput(ybase, xmax, ylen, zbase)))
			addFin(i+1, ybase+ylen, zbase-0.5*zlen, zbase+0.5*zlen, xlen);
	}

	var addBody = function(i, rearz, prev) {
		// Gen new body segment
		var rets = addBodySeg(rearz, prev);
		var xlen = rets.xlen;
		var ylen = rets.ylen;
		var zlen = rets.zlen;
		var bodyType = rets.type;
		// Gen wings?
		var wingprob = wi(i+1, 0.5);
		// if (_flip(wingprob))
		if (_flip(wingprob, nnInput(rearz, xlen, ylen, zlen)))
			addWings(0, 0.5*xlen, rearz+0.5, rearz+zlen-0.5);
		// Gen fin?
		var finprob = 0.7;
		// if (_flip(finprob))
		if (_flip(finprob, nnInput(rearz, xlen, ylen, zlen)))
			addFin(0, 0.5*ylen, rearz, rearz+zlen, 0.6*xlen);
		// Continue generating?
		var nextprev = {type: bodyType, xlen: xlen, ylen: ylen};
		// if (_flip(wi(i, 0.4)))
		if (_flip(wi(i, 0.4), nnInput(rearz, xlen, ylen, zlen)))
			addBody(i+1, rearz+zlen, nextprev);
		else {	
			// TODO: Also have a box nose, like the old version?
			// if (_flip(0.75))
			if (_flip(0.75, nnInput(rearz, xlen, ylen, zlen)))
				addCylinderBodySeg(rearz+zlen, nextprev, true);
		}
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

	var numIntersections = function(geolist) {
		var n = 0;
		for (var i = 0; i < geolist.length-1; i++) {
			for (var j = i+1; j < geolist.length; j++) {
				if (geolist[i].intersects(geolist[j])) {
					n++;
				}
			}
		}
		return n;
	};

	// ----------------------------------------------------------------------------

	// This is the 'main' function of the program
	var generate = function(params) {
		prm = function(v, t, s, h) { return param(params, v, t, s, h); };

		globalStore.geometry = [];
		globalStore.volume = 0;
		addBody(0, -5, {type: null, xlen: 0, ylen: 0});

		factorFunc(function() {
			var f = 0;

			// Encourage desired aspect ratio
			var bbox = new THREE.Box3();
			for (var i = 0; i < globalStore.geometry.length; i++)
				bbox.union(globalStore.geometry[i].getbbox());
			var size = bbox.size();
			// var targetWidth = 10;
			// var targetLength = 10;
			var targetWidth = 5;
			var targetLength = 15;
			f += gaussFactor(size.x, targetWidth, 0.1);
			f += gaussFactor(size.z, targetLength, 0.1);

			// // Encourage desired volume
			// var targetVolume = 60;
			// f += gaussFactor(globalStore.volume, targetVolume, 0.1);

			// // Discourage self-intersection
			// var nisects = numIntersections(globalStore.geometry);
			// f += gaussFactor(nisects, 0, 0.1);

			return f;
		});

		return globalStore.geometry;
	}

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
				console.log('address: ' + address);
				throw 'ERP param neural net has no inputs!';
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
// var name = 'spacehip_bbox';
// var name = 'spacehip_isect';
// var name = 'spacehip_bbox+isect';

// Mean field variational test
var target = makeProgram('target');
// var guide = makeProgram('meanField');
var guide = makeProgram('neural');
var result = variational.infer(target, guide, undefined, {
	verbosity: 3,
	nSamples: 100,
	nSteps: 100,
	convergeEps: 0.1,
	initLearnrate: 0.5
	// initLearnrate: 0.1
});
variational.saveParams(result.params, 'procmod/results/'+name+'.params');
// var result = { params: variational.loadParams('procmod/results/'+name+'.params') };
var geos = [];
for (var i = 0; i < 10; i++) {
	var geolist = util.runWithAddress(guide, '', [result.params]);
	// var geolist = util.runWithAddress(target, '');
	var combgeo = Geo.mergeGeometries(geolist);
	var bboxsize = combgeo.getbbox().size()
	console.log(bboxsize.x, bboxsize.z);
	geos.push(combgeo);
}
require('procmod/lib/utils').saveLineup(geos, 'procmod/results/'+name+'.obj');






