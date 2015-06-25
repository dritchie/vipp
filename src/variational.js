'use strict';

var numeric = require('numeric');


// Get the primal value of a dual number/tape
function primal(x) {
	return x.primal === undefined ? x : x.primal;
}


// Global inference coroutine
var coroutine = {
	paramIndex: 0,
	sample: function(erp, params) {
		return erp.sample(params);
	},
	factor: function(num) {}
};

// Run a variational program with a given param set
// (This is just needed to track the param index)
function run(guide, args, params) {
	coroutine.paramIndex = 0;
	return guide(args, params);
}

// target: original probabilistic program
// guide: variational program
// args: inputs to program (i.e. observed evidence)
// opts: options controlling inference behavior
// Returns the inferred variational params
function infer(target, guide, args, opts) {
	// Extract options
	opts = opts || {};
	function opt(val, defaultval) {
		return val === undefined ? defaultval : val;
	}
	var nSteps = opt(opts.nSteps, 100);
	var nSamples = opt(opts.nSamples, 100);
	var learnRate = opt(opts.initLearnRate, 1);
	var convergeEps = opt(opts.convergeEps, 0.1);
	var verbosity = opt(opts.verbosity, 0);

	// Define the inference coroutine
	var vco = {
		sample: function(erp, params) {
			var val = this.choices[this.choices.length-1];
			if (val === undefined) {
				// We don't store tapes in the trace, just raw numbers, so that
				//    re-running with the target program works correctly.
				var pparams = params.map(function(x) { return primal(x); });
				val = erp.sample(pparams);
				this.choices.push(val);
			}
			// console.log(params.map(function(x) { return primal(x); }), primal(val), primal(erp.score(params, val)));
			this.score = ad_add(this.score, erp.score(params, val));
			return val;
		},
		factor: function(num) {
			this.score = ad_add(this.score, num);
		},
		run: function(thunk) {
			this.paramIndex = 0;
			this.choices = [];
			return this.rerun(thunk);
		},
		rerun: function(thunk) {
			this.paramIndex = 0;
			this.choiceIndex = 0;
			this.score = 0;	
			return thunk();
		}
	}

	// Install coroutine
	var oldCoroutine = coroutine;
	coroutine = vco;

	var params = [];

	// Thunks that we'll feed to 'run' and 'rerun'
	var targetThunk = function() {
		return target(args);
	}
	var guideThunk = function() {
		return guide(params, args);
	}
	var guideGrad = ad_gradientR(function(p) {
		guide(p, args);
		return vco.score;
	})
	var guideGradThunk = function() {
		return guideGrad(params);
	}

	
	// Run guide once to initialize vector of params
	// TODO: This will not work if params has variable size--we'll need to
	//    adopt a different strategy then.
	vco.run(guideThunk);
	// Do variational inference
	var currStep = 0;
	var maxDeltaAvg = 0;
	var runningG2 = numeric.rep([params.length], 0);
	do {
		if (verbosity > 1)
			console.log('Variational iteration ' + (currStep+1) + '/' + nSteps);
		if (verbosity > 2)
			console.log('  params: ' + params.toString());
		// Estimate learning signal with guide samples
		var sumGrad = numeric.rep([params.length], 0);
		var sumGradSq = numeric.rep([params.length], 0);
		var sumWeightedGrad = numeric.rep([params.length], 0);
		var sumWeightedGradSq = numeric.rep([params.length], 0);
		for (var s = 0; s < nSamples; s++) {
			if (verbosity > 3)
				console.log('  Sample ' + s + '/' + nSamples);
			var grad = vco.run(guideGradThunk);
			var guideScore = vco.score.primal;
			vco.rerun(targetThunk);
			var targetScore = vco.score;
			var scoreDiff = targetScore - guideScore;
			var weightedGrad = numeric.mul(grad, scoreDiff);
			if (verbosity > 4) {
				console.log('    guide score: ' + guideScore + ', target score: ' + targetScore
					+ ', diff: ' + (targetScore - guideScore));
				console.log('    grad: ' + grad.toString());
				console.log('    weightedGrad: ' + weightedGrad.toString());
			}
			// throw "early out";
			numeric.addeq(sumGrad, grad);
			numeric.addeq(sumWeightedGrad, weightedGrad);
			numeric.poweq(grad, 2);	// grad is now gradSq
			numeric.addeq(sumGradSq, grad);
			var weightedGradSq = numeric.mul(grad, scoreDiff)
			numeric.addeq(sumWeightedGradSq, weightedGradSq);
		}
		// Compute AdaGrad learning rate and control variate,
		//    then do parameter update
		var aStar = numeric.div(sumWeightedGradSq, sumGradSq);
		numeric.muleq(aStar, sumGrad);
		var elboGradEst = numeric.sub(sumWeightedGrad, aStar);
		numeric.diveq(elboGradEst, nSamples);
		if (verbosity > 2) {
			console.log('  sumGrad: ' +  sumGrad.toString());
			console.log('  sumWeightedGrad: ' +  sumWeightedGrad.toString());
			console.log('  elboGradEst: ' +  elboGradEst.toString());
		}
		// if (currStep == 1)
		// 	throw "early out";
		var maxDelta = 0;
		for (var i = 0; i < params.length; i++) {
			var grad = elboGradEst[i];
			runningG2[i] += grad*grad;
			var weight = learnRate / Math.sqrt(runningG2[i]);
			var delta = weight * grad;
			params[i] += delta;
			maxDelta = Math.max(Math.abs(delta), maxDelta);
		}
		// Check for convergence
		maxDeltaAvg = maxDeltaAvg * 0.9 + maxDelta;
		var converged = maxDeltaAvg < convergeEps;
		currStep++;
	} while (!converged && currStep < nSteps);
	if (verbosity > 0) {
		if (converged)
			console.log('CONVERGED after step ' + currStep + ' (' + maxDeltaAvg + ' < ' + convergeEps + ')');
		else
			console.log('DID NOT CONVERGE (' + maxDeltaAvg + ' > ' + convergeEps + ')');
	}

	// Restore original coroutine
	oldCoroutine = oldCoroutine;

	return params;
}

function sample(erp, params) {
	return coroutine.sample(erp, params);
}

function factor(num) {
	coroutine.factor(num);
}

// Create/lookup a param
function param(params, initialVal) {
	if (coroutine.paramIndex == params.length)
		params.push(primal(initialVal));
	var ret = params[coroutine.paramIndex];
	coroutine.paramIndex++;
	return ret;
}

// Create/lookup a param that has a prior
function paramWithPrior(params, initialVal, scoreFn, hypers) {
	var p = param(params, initialVal);
	factor(scoreFn(hypers, p));
	return p;
}

module.exports = {
	run: run,
	infer: infer,
	sample: sample,
	factor: factor,
	param: param,
	paramWithPrior: paramWithPrior
};



