var THREE = require('three');
var Geo = require('procmod/lib/geometry');
var SpaceshipUtil = require('procmod/programs/spaceshipUtil');
var bounds = require('src/boundsTransforms');
var util = require('src/util');


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

	var weightedSum = function(nums, i) {
		if (i === undefined) i = 0;
		return i === nums.length ? 0 : prm(0) * nums[i] + weightedSum(nums, i + 1);
	}

	// Possible layer transforms
	var id = function(x) { return x; };
	var neg_exp = function(x) { return Math.exp(-x); };
	var sigmoid = function(x) { return 1 / (1 + Math.exp(-x)); };
	var tanh = function(x) { return Math.tanh(x); };

	var nnLayer = function(inputs, n, transform) {
		return repeat(n, function() {
			return transform(weightedSum(inputs));
		});
	}

	// Single-layer perceptron
	var perceptron = function(inputs, nHidden, nOut, outTransform) {
		// TODO: standardize/rectify input, somehow?
		var hiddenLayer = nnLayer(inputs, nHidden, tanh);
		return nnLayer(hiddenLayer, nOut, outTransform);
	}

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
		var ERP_NHIDDEN = 1;
		erpGetParams = function(params, bounds, nninputs, outTransform) {
			// For now, neural nets share parameters per callsite
			var address = arguments[0];
			var callsite = address.slice(address.lastIndexOf('_', address.lastIndexOf('_')));
			return util.runWithAddress(perceptron, callsite, [nninputs, ERP_NHIDDEN, params.length, outTransform]);
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
		var xl = _uniform(1, 3);
		var yl = _uniform(.5, 1) * xl;
		var xlen = (prev.type === BodyType.Box) ? xl : Math.max(xl, prev.xlen);
		var ylen = (prev.type === BodyType.Box) ? yl : Math.max(yl, prev.ylen);
		var zlen = _uniform(2, 5);
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
		var radius = _uniform(minrad, maxrad);
		var xlen = radius*2;
		var ylen = radius*2;
		var zlen = isnose ? _uniform(1, 3) : _uniform(2, 5);
		var geo = isnose ? SpaceshipUtil.BodyCylinder(rearz, zlen, radius,
													 radius*_uniform(.25, .75))
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
		var radius = _uniform(minrad, maxrad);
		var xlen = radius*4;
		var ylen = radius*4;
		var zlen = _uniform(2, 5);
		var geo = SpaceshipUtil.BodyCluster(rearz, zlen, radius);
		addGeometry(geo);
		addVolume(radius*radius*Math.PI*zlen*4);
		return { xlen: xlen, ylen: ylen, zlen: zlen, type: BodyType.Cluster };
	}

	var BodyType = { Box: 0, Cylinder: 1, Cluster: 2, N: 3 }
	var addBodySeg = function(rearz, prev) {	
		// var type = randomInteger(BodyType.N);
		var type = _discrete([.33, .33, .33]);
		if (type == BodyType.Box)
			return addBoxBodySeg(rearz, prev)
		else if (type == BodyType.Cylinder)
			return addCylinderBodySeg(rearz, prev);
		else if (type == BodyType.Cluster)
			return addClusterBodySeg(rearz, prev);
		else throw('unsupported body type ' + type);
	}

	var addBoxWingSeg = function(xbase, zlo, zhi) {
		var zbase = _uniform(zlo, zhi);
		var xlen = _uniform(0.25, 2.0);
		var ylen = _uniform(0.25, 1.25);
		var zlen = _uniform(0.5, 4.0);
		var geo = SpaceshipUtil.WingBoxes(xbase, zbase, xlen, ylen, zlen);
		addGeometry(geo);
		addVolume(xlen*ylen*zlen*2);
		if (_flip(0.5))
			addWingGuns(xbase, zbase, xlen, ylen, zlen);
		return { xlen: xlen, ylen: ylen, zlen: zlen, zbase: zbase };
	}

	var addWingGuns = function(xbase, zbase, xlen, ylen, zlen) {
		var gunlen = _uniform(1, 1.2)*zlen;
		var gunxbase = xbase + 0.5*xlen;
		var gunybase = 0.5*ylen;
		var geo = SpaceshipUtil.WingGuns(gunxbase, gunybase, zbase, gunlen);
		addGeometry(geo);
		// Let's just say that guns don't count for overall volume...
	};

	var addCylinderWingSeg = function(xbase, zlo, zhi) {
		var zbase = _uniform(zlo, zhi);
		var radius = _uniform(.15, .7);
		var xlen = 2*radius;
		var ylen = 2*radius;
		var zlen = _uniform(1, 5);
		var geo = SpaceshipUtil.WingCylinders(xbase, zbase, zlen, radius);
		addGeometry(geo);
		addVolume(radius*radius*Math.PI*zlen*2);
		return { xlen: xlen, ylen: ylen, zlen: zlen, zbase: zbase };
	}

	var WingType = { Box: 0, Cylinder: 1, N: 2 }
	var addWingSeg = function(xbase, zlo, zhi) {
		// var type = randomInteger(WingType.N);
		var type = _flip(0.5) + 0;
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
		if (_flip(wi(i, 0.6)))
			addWings(i+1, xbase+xlen, zbase-0.5*zlen, zbase+0.5*zlen);
	}

	var addFin = function(i, ybase, zlo, zhi, xmax) {
		var xlen = _uniform(0.5, 1.0) * xmax;
		var ylen = _uniform(0.1, 0.5);
		var zlen = _uniform(0.5, 1.0) * (zhi - zlo);
		var zbase = 0.5*(zlo + zhi);
		var geo = Geo.Shapes.Box(0, ybase + 0.5*ylen, zbase, xlen, ylen, zlen);
		addGeometry(geo);
		addVolume(xlen*ylen*zlen);
		if (_flip(wi(i, 0.2)))
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
		if (_flip(wingprob))
			addWings(0, 0.5*xlen, rearz+0.5, rearz+zlen-0.5);
		// Gen fin?
		var finprob = 0.7;
		if (_flip(finprob))
			addFin(0, 0.5*ylen, rearz, rearz+zlen, 0.6*xlen);
		// Continue generating?
		var nextprev = {type: bodyType, xlen: xlen, ylen: ylen};
		if (_flip(wi(i, 0.4)))
			addBody(i+1, rearz+zlen, nextprev);
		else {	
			// TODO: Also have a box nose, like the old version?
			if (_flip(0.75))
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

			// Encourage desired volume
			var targetVolume = 60;
			f += gaussFactor(globalStore.volume, targetVolume, 0.1);

			// // Discourage self-intersection
			// var nisects = numIntersections(globalStore.geometry);
			// f += gaussFactor(nisects, 0, 0.1);

			return f;
		});

		return globalStore.geometry;
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
var guide = makeProgram('meanField');
var result = variational.infer(target, guide, undefined, {
	verbosity: 3,
	nSamples: 100,
	nSteps: 200,
	convergeEps: 0.1,
	initLearnrate: 0.5
});
variational.saveParams(result.params, 'procmod/results/'+name+'.params');
// var result = { params: variational.loadParams('procmod/results/'+name+'.params') };
var geos = [];
for (var i = 0; i < 10; i++) {
	var geolist = util.runWithAddress(guide, '', [result.params]);
	// var geolist = util.runWithAddress(target, '');
	geos.push(Geo.mergeGeometries(geolist));
}
require('procmod/lib/utils').saveLineup(geos, 'procmod/results/'+name+'.obj');






