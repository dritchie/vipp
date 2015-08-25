
var THREE = require('THREE');
var TreeUtils = require('procmod/programs/treeUtil');
var bounds = require('src/boundsTransforms');
var Geo = require('procmod/lib/geometry');

var makeProgram = function(isGuide) {

	var globalStore = {};

	// ----------------------------------------------------------------------------

	// Parameter function
	var prm;

	var _gaussian = function(mu, sigma) {
		return gaussian(prm(mu), prm(sigma, bounds.nonNegative));
	}

	var _uniform = function(lo, hi) {
		// var u = uniform(prm(0), prm(1));		// Doesn't work
		var u = beta(prm(1.0, bounds.nonNegative), prm(1.0, bounds.nonNegative));
		return (1.0-u)*lo + u*hi;
	}

	var _flip = function(p) {
		p = Math.min(Math.max(0.01, p), 0.99);  // Can't be exactly 0 or exactly 1
		return flip(prm(p, bounds.unitInterval));
	}

	// ----------------------------------------------------------------------------

	var continueProb = function(depth) {
		return Math.exp(-0.1*depth);
	};

	var lerp = function(lo, hi, t) { return (1-t)*lo + t*hi; };
	var clamp = function(x, lo, hi) { return Math.min(Math.max(x, lo), hi); };

	var maxrotamt = Math.PI/7;
	var worldup = new THREE.Vector3(0, 1, 0);
	var projworldup = new THREE.Vector3();
	var tmp = new THREE.Vector3();
	var samplePlaneRot = function(fwd, normal) {
		projworldup.copy(worldup).projectOnPlane(normal);
		var dot = projworldup.dot(fwd);
		tmp.copy(projworldup).cross(fwd);
		var neg = normal.dot(tmp) > 0;
		var lobound = neg ? -maxrotamt : lerp(0, -maxrotamt, Math.max(dot, 0));
		var hibound = neg ? lerp(0, maxrotamt, Math.max(dot, 0)) : maxrotamt;
		return _uniform(lobound, hibound);
	}

	var left = new THREE.Vector3();
	var branch = function(r0, curr, i, d, prev) {
		// Stop generating if branches get too small
		if (curr.radius / r0 >= 0.1) {
			// Sample rotation change for new branch segment
			// var uprot = _gaussian(0, Math.PI / 12);
			// var leftrot = _gaussian(0, Math.PI / 12);
			// var uprot = _uniform(-Math.PI/7, Math.PI/7);
			// var leftrot = _uniform(-Math.PI/7, Math.PI/7);
			var uprot = samplePlaneRot(curr.forward, curr.up);
			left.copy(curr.up).cross(curr.forward);
			var leftrot = samplePlaneRot(curr.forward, left);

			// Sample length and radius of new branch segment
			var len = _uniform(3, 5) * curr.radius;
			var endradius = _uniform(0.7, 0.9) * curr.radius;

			// Tree segments represented by two connected conic sections
			var next = TreeUtils.advanceFrame(curr, uprot, leftrot, len, endradius);
			var split = TreeUtils.findSplitFrame(curr, next);
			var geom = TreeUtils.treeSegment(prev, curr, split, next);
			globalStore.geometry.push(geom);

			// Recursively branch?
			// var branchprob = 0.5;
			var branchprob = lerp(0, 0.6, Math.min(1, curr.center.y/5));
			if (_flip(branchprob)) {
				// Branches more likely on upward-facing parts of parent branch
				var upnessDistrib = TreeUtils.estimateUpness(split, next);
				var theta = _gaussian(upnessDistrib[0], upnessDistrib[1]);
				var branchradius = _uniform(0.9, 1) * endradius;
				// Sample where along the parent segment this branch should occur
				// var t = 0.5;
				var tpadding = 0.05;
				var effectivelen = tmp.copy(next.center).sub(split.center).length();
				var tlobound = clamp(0.5*branchradius/effectivelen + tpadding, tpadding, 1-tpadding);
				var thibound = clamp((effectivelen - 0.5*branchradius)/effectivelen, tpadding, 1-tpadding);
				var t = _uniform(tlobound, thibound);
				var b = TreeUtils.branchFrame(split, next, t, theta, branchradius);
				branch(r0, b.frame, 0, d + 1, b.prev);
			}

			// // Keep generating same branch?
			// if (_flip(continueProb(i)))
				branch(r0, next, i + 1, d, null);
		}
	};

	// ----------------------------------------------------------------------------

	var factorFunc;
	if (!isGuide) {
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

	var generate = function(params) {

		if (isGuide)
			prm = function(v, t, s, h) { return param(params, v, t, s, h); };
		else
			prm = function(x) { return x; };

		globalStore.geometry = [];

		var start = {
			center: new THREE.Vector3(0, 0, 0),
			forward: new THREE.Vector3(0, 1, 0),
			up: new THREE.Vector3(0, 0, -1),
			radius: _uniform(1.5, 2),
			v: 0
		};

		branch(start.radius, start, 0, 0, null);

		factorFunc(function() {
			var f = 0;

			// Encourage desired aspect ratio
			var bbox = new THREE.Box3();
			for (var i = 0; i < globalStore.geometry.length; i++)
				bbox.union(globalStore.geometry[i].getbbox());
			var size = bbox.size();
			var targetWidth = 15;
			var targetLength = 15;
			var targetHeight = 40;
			f += gaussFactor(size.x, targetWidth, 0.2);
			f += gaussFactor(size.z, targetLength, 0.2);
			f += gaussFactor(size.y, targetHeight, 0.2);

			// Discourage self-intersection
			var nisects = numIntersections(globalStore.geometry);
			f += gaussFactor(nisects, 0, 0.1);

			return f;
		});

		return globalStore.geometry;
	};

	return generate;

}

// ----------------------------------------------------------------------------


var name = 'test';
// var name = 'tree_bbox';
// var name = 'tree_isect';
// var name = 'tree_bbox+isect';

// Mean field variational test
var target = makeProgram(false);
var guide = makeProgram(true);
var result = variational.infer(target, guide, undefined, {
	verbosity: 3,
	nSamples: 100,
	nSteps: 100,
	convergeEps: 0.1,
	initLearnrate: 0.5
});
variational.saveParams(result.params, 'procmod/results/'+name+'.params');
// var result = { params: variational.loadParams('procmod/results/'+name+'.params') };
var util = require('src/util');
var procmodUtils = require('procmod/lib/utils');
var geos = [];
for (var i = 0; i < 10; i++) {
	var geolist = util.runWithAddress(guide, '', [result.params]);
	// var geolist = util.runWithAddress(target, '');
	geos.push(Geo.mergeGeometries(geolist));
}
procmodUtils.saveLineup(geos, 'procmod/results/'+name+'.obj');






