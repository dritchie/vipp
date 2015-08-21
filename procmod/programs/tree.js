
var THREE = require('THREE');
var TreeUtils = require('procmod/programs/treeUtil');
var bounds = require('src/boundsTransforms');

var makeProgram = function(isGuide) {

	var globalStore = {};

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

	var continueProb = function(depth) {
		return Math.exp(-0.1*depth);
	};

	var branch = function(r0, curr, i, d, prev) {
		// Stop generating if branches get too small
		if (curr.radius / r0 >= 0.1) {
			var uprot = _gaussian(0, Math.PI / 12);
			var leftrot = _gaussian(0, Math.PI / 12);
			var len = _uniform(3, 5) * curr.radius;
			var endradius = _uniform(0.7, 0.9) * curr.radius;

			// Tree segments represented by two connected conic sections
			var next = TreeUtils.advanceFrame(curr, uprot, leftrot, len, endradius);
			var split = TreeUtils.findSplitFrame(curr, next);
			var geom = TreeUtils.treeSegment(prev, curr, split, next);
			globalStore.geometry = globalStore.geometry.concat([geom]);

			// Recursively branch?
			if (_flip(0.5)) {
				// Branches more likely on upward-facing parts of parent branch
				var upnessDistrib = TreeUtils.estimateUpness(split, next);
				var theta = _gaussian(upnessDistrib[0], upnessDistrib[1]);
				var branchradius = _uniform(0.9, 1) * endradius;
				// Branches spawn in middle of parent branch
				var t = 0.5;
				var b = TreeUtils.branchFrame(split, next, t, theta, branchradius);
				branch(r0, b.frame, 0, d + 1, b.prev);
			}

			// Keep generating same branch?
			if (_flip(continueProb(i)))
				branch(r0, next, i + 1, d, null);
		}
	};


	var genTree = function() {

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

		return globalStore.geometry;
	};

	return genTree;

}

// Forward sampling test
var generate = makeProgram(false);
var geolist = generate();
var accumgeo = require('procmod/lib/geometry').mergeGeometries(geolist);
require('procmod/lib/utils').saveOBJ(accumgeo, 'test.obj');



