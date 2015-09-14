var utils = require('src/util');
var lutils = require('lsystem/utils');
var THREE = require('three');
var bounds = require('src/boundsTransforms');
var nodeutil = require('util');


var mapIndexed = function(f, l) {
  var fn = function(ls, i, _acc) {
    return ls.length === 0 ?
        _acc :
        fn(ls.slice(1), i + 1, _acc.concat([f(i, ls[0])]))
  };
  return fn(l, 0, []);
}

var polar2rect = function(r, theta) {
	return new THREE.Vector2(r*Math.cos(theta), r*Math.sin(theta));
};




var mkbranch = function(start, angle, width, length) {
	return {
		start: start,
		angle: angle,
		width: width,
		end: start.clone().add(polar2rect(length, angle))
	};
};



// var loresW = 50;
// var loresH = 50;
var loresW = 100;
var loresH = 100;
var canvas = lutils.newCanvas(loresW, loresH);
var viewport = {xmin: -20, xmax: 20, ymin: -38, ymax: 2};

var makeProgram = function(opts) {

	var family = opts.family;

	var globals = {};

	var prm = function(v, t, s, h) { return param(globals.params, v, t, s, h); };

	var getParams;
	if (family === 'target')
		getParams = function(params) { return params; };
	else if (family === 'meanField') {
		getParams = function(params, bounds) {
			return mapIndexed(function(i, p) { return prm(params[i], bounds[i]); }, params);
		};
	}

	var _gaussian = function(mu, sigma) {
		var address = arguments[0];
		var addrParts = address.split('_');
		var callsite = '_' + addrParts[addrParts.length-1];
		// var params = getParams([mu, sigma], [undefined, bounds.nonNegative]);
		var params = utils.runWithAddress(getParams, callsite, [[mu, sigma], [undefined, bounds.nonNegative]]);
		return gaussian(params[0], params[1]);
	};

	var _flip = function(p) {
		// p = Math.min(Math.max(0.001, p), 0.999);  // Can't be exactly 0 or exactly 1
		// var address = arguments[0];
		// var addrParts = address.split('_');
		// var callsite = '_' + addrParts[addrParts.length-1];
		// // var params = getParams([p], [bounds.unitInterval]);
		// var params = utils.runWithAddress(getParams, callsite, [[p], [bounds.unitInterval]]);
		// return flip(params[0]);
		return flip(p);
	}

	var factorFunc;
	if (family === 'target') {
		factorFunc = function(fn) { factor(fn()); };
	} else factorFunc = function() {};

	// ------------------------------------------------------------------------

	var branch = function(depth, currState, branches) {
		var width = 0.9 * currState.width;
		var length = 2;
		var newang = currState.angle + _gaussian(0, Math.PI/8);
		var newbranch = mkbranch(currState.pos, newang, width, length);
		branches.push(newbranch);
		// Terminate?
		if (_flip(Math.exp(-0.045*depth))) {
			// Continue or fork?
			if (_flip(0.5)) {
				branch(depth + 1, {pos: newbranch.end, angle: newbranch.angle, width: newbranch.width}, branches);
			} else {
				var branchState = {
					pos: newbranch.end,
					angle: newbranch.angle - Math.abs(_gaussian(0, Math.PI/6)),
					width: newbranch.width
				};
				branch(depth + 1, branchState, branches);
				branchState.angle = newbranch.angle + Math.abs(_gaussian(0, Math.PI/6));
				branch(depth + 1, branchState, branches);
			}
		}
	};

	var generate = function(params) {
		globals.params = params;
		var branches = [];
		var startState = {
			pos: new THREE.Vector2(0, 0),
			angle: -Math.PI/2,
			width: 0.75
		}
		branch(0, startState, branches);

		factorFunc(function() {
			var f = 0;

			lutils.render(canvas, viewport, branches);
			var img2d = lutils.newImageData2D(canvas);

			// Horizonal bilateral symmetry
			var sym = img2d.filledBilateralSymmetryHoriz();
			f += gaussianERP.score([1, 0.01], sym);

			return f;
		});

		return branches;
	};

	return generate;
};


var target = makeProgram({family: 'target'});
var guide =  makeProgram({family: 'meanField'});

var result = variational.infer(target, guide, undefined, {
	verbosity: 3,
	nSamples: 100,
	nSteps: 200,
	convergeEps: 0.1,
	initLearnrate: 1,

	// tempSchedule: lutils.TempSchedules.linearStop(0.5)
	// tempSchedule: lutils.TempSchedules.asymptotic(10)
});

var outname = 'test';
var fs = require('fs');
var cp = require('child_process');
var dirname = 'lsystem/results/' + outname;
if (fs.existsSync(dirname))
	cp.execSync('rm -rf ' + dirname);
fs.mkdirSync(dirname);
for (var i = 0; i < 20; i++) {
	var branches = utils.runWithAddress(guide, '', [result.params]);
	// var branches = utils.runWithAddress(target, '');
	lutils.render(canvas, viewport, branches);
	var img2d = lutils.newImageData2D(canvas);
	var sym = img2d.filledBilateralSymmetryHoriz();
	lutils.renderOut(nodeutil.format('%s/%d_%d.png', dirname, i, sym),
		{width: 600, height: 600}, viewport, branches);
}


// // TEST
// var generate = makeProgram({family: 'target'});
// var branches = generate();
// console.log(branches.length);
// var bbox = new THREE.Box2();
// for (var i = 0; i < branches.length; i++) {
// 	var br = branches[i];
// 	bbox.expandByPoint(br.start);
// 	bbox.expandByPoint(br.end);
// }
// console.log(bbox);
// lutils.renderOut(
// 	'lsystem/results/lsystem.png',
// 	{width: 600, height: 600},
// 	{xmin: -20, xmax: 20, ymin: -38, ymax: 2},
// 	branches
// );






