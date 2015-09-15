var utils = require('src/util');
var lutils = require('lsystem/utils');
var THREE = require('three');
var bounds = require('src/boundsTransforms');
var nodeutil = require('util');
var _ = require('underscore');



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
	// Neural starts out doing the same thing as target (while it's learning how to
	//    normalize the state and tree node data)
	if (family === 'target' || family === 'neural') {
		getParams = function(params) { return params; };
	} else if (family === 'meanField') {
		getParams = function(params, bounds) {
			var address = arguments[0];
			var addrParts = address.split();
			var callsite = '_' + addrParts[addrParts.length-2];
			var prms = [];
			for (var i = 0; i < params.length; i++)
				prms.push(utils.runWithAddress(prm, callsite+'['+i+']', [params[i], bounds[i]]));
			return prms;
		};
	}

	var collectInfo = function() {};
	if (family === 'neural') {
		var rnn = require('src/neuralnet/rnn');
		var rnnopts = _.extend(_.clone(opts), {globals: globals});
		var paramPredictor = rnn.makeParamPredictor(rnnopts);
		collectInfo = function(currState, treeNode) {
			paramPredictor.collectStateSample(currState);
			paramPredictor.collectTreeNodeSample(treeNode);
		}
	}

	var _gaussian = function(mu, sigma) {
		var params = getParams([mu, sigma], [bounds.none, bounds.nonNegative], globals.currState);
		return gaussian(params[0], params[1]);
	};

	// TODO: Also parameterize flips?
	var _flip = function(p) {
		return flip(p);
	}

	var factorFunc;
	if (family === 'target') {
		factorFunc = function(fn) { factor(fn()); };
	} else factorFunc = function() {};

	// ------------------------------------------------------------------------

	var polar2rect = function(r, theta) {
		return new THREE.Vector2(r*Math.cos(theta), r*Math.sin(theta));
	};

	var branch = function(currState, treeNode) {
		globals.currState = currState;
		var width = 0.9 * currState.width;
		var length = 2;
		var newang = currState.angle + _gaussian(0, Math.PI/8);
		var newbranch = {
			start: currState.pos,
			angle: newang,
			width: width,
			end: currState.pos.clone().add(polar2rect(length, newang))
		};
		var newNode = { branch: newbranch, children: [] };
		if (treeNode === undefined) {
			globals.treeRoot = newNode;
		} else {
			treeNode.children.push(newNode);
		}
		globals.branches.push(newbranch);
		collectInfo(currState, newNode);
		// Terminate?
		if (_flip(Math.exp(-0.045*currState.depth))) {
			// Continue or fork?
			if (_flip(0.5)) {
				branch({
					depth: currState.depth + 1,
					pos: newbranch.end,
					angle: newbranch.angle,
					width: newbranch.width},
				newNode);
			} else {
				var branchState = {
					depth: currState.depth + 1,
					pos: newbranch.end,
					angle: newbranch.angle - Math.abs(_gaussian(0, Math.PI/6)),
					width: newbranch.width
				};
				branch(branchState, newNode);
				branchState.angle = newbranch.angle + Math.abs(_gaussian(0, Math.PI/6));
				branch(branchState, newNode);
			}
		}
	};

	var generate = function(params) {
		globals.params = params;
		globals.treeRoot = undefined;
		globals.branches = [];
		var startState = {
			depth: 0,
			pos: new THREE.Vector2(0, 0),
			angle: -Math.PI/2,
			width: 0.75
		}
		branch(startState, undefined);

		factorFunc(function() {
			var f = 0;

			lutils.render(canvas, viewport, globals.branches);
			var img2d = lutils.newImageData2D(canvas);

			// Horizonal bilateral symmetry
			var sym = img2d.filledBilateralSymmetryHoriz();
			f += gaussianERP.score([1, 0.01], sym);

			return f;
		});

		return globals.branches;
	};

	// If using the neural family, run generate until the param predictor has learned how
	//    to normalize the state and tree node data
	if (family === 'neural') {
		do {
			generate();
		}
		while (!paramPredictor.readyToPredict())
		// Now, replace 'getParams' with the neural net version
		getParams = function(params, bounds, currState) {
			var address = arguments[0];
			var addrParts = address.split();
			var callsite = '_' + addrParts[addrParts.length-2];
			return paramPredictor.predict(callsite, bounds, currState, globals.treeRoot);
		};
		collectInfo = function() {};
	}

	return generate;
};


var target = makeProgram({family: 'target'});
// var guide =  makeProgram({family: 'meanField'});
var guide = makeProgram({family: 'neural',
	stateFeatures: lutils.FeatureExtractors.state,
	treeNodeFeatures: lutils.FeatureExtractors.treeNode,
	latentN: 10,
	nNormalizeSamples: 1000
})

var result = variational.infer(target, guide, undefined, {
	verbosity: 4,
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






