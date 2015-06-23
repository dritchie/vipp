'use strict';

var numeric = require('numeric');


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

	// Define the inference coroutine
	var vco = {
		sample: function(erp, params) {
			if (this.choiceIndex === this.choices.length) {
				val = erp.sample(params);
				this.score = ad_add(this.score, erp.score(params, val));
				this.choices.push(val);
			}
			return this.choices[this.choices.length-1];
		},
		factor: function(num) {
			this.score = ad_add(this.score, num);
		},
		run: function(thunk) {
			this.choices = [];
			return this.rerun(thunk);
		},
		rerun: function(thunk) {
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
	var guideGrad = ad_gradientR(function(p) {
		guide(p, args);
		return vco.score;
	})
	var guideGradThunk = function() {
		return guideGrad(params);
	}

	// Do variational inference
	var currStep = 0;
	var runningG2 = null;
	var maxDeltaAvg = 0;
	do {
		// Estimate learning signal with guide samples
		var sumGrad = null;
		var sumGradSq = null;
		var sumWeightedGrad = null;
		var sumWeightedGradSq = null;
		for (var s = 0; s < nSamples; s++) {
			var grad = vco.run(guideGradThunk);
			var guideScore = vco.score.primal;
			vco.rerun(targetThunk);
			var targetScore = vco.score;
			var weightedGrad = numeric.mul(grad, targetScore - guideScore);
			if (sumGrad === null) {
				sumGrad = numeric.rep([grad.length], 0);
				sumGradSq = numeric.rep([grad.length], 0);
				sumWeightedGrad = numeric.rep([grad.length], 0);
				sumWeightedGradSq = numeric.rep([grad.length], 0);
			}
			numeric.addeq(sumGrad, grad);
			numeric.addeq(sumWeightedGrad, weightedGrad);
			numeric.poweq(grad, 2);
			numeric.poweq(weightedGrad, 2);
			numeric.addeq(sumGradSq, grad);
			numeric.addeq(sumWeightedGradSq, weightedGrad);
		}
		// Compute AdaGrad learning rate and control variate,
		//    then do parameter update
		var aStar = numeric.div(sumWeightedGradSq, sumGradSq);
		numeric.muleq(aStar, sumGrad);
		var elboGradEst = numeric.sub(sumWeightedGrad, aStar);
		numeric.diveq(elboGradEst, nSamples);
		if (runningG2 === null) runningG2 = numeric.rep([params.length], 0);
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

	// Restore original coroutine
	oldCoroutine = oldCoroutine;

	return params;
}

function sample(erp, params) {
	return coroutine.sample(erp, params);
}

function factor(num) {
	return coroutine.factor(num);
}

// Create/lookup a param
function param(params, initialVal) {
	if (coroutine.paramIndex == params.length)
		params.push(initialVal);
	return params[params.length-1];
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



