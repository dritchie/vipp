'use strict';

var numeric = require('numeric');
var present = require('present');

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
	var learnRate = opt(opts.initLearnRate, 0.5);
	var convergeEps = opt(opts.convergeEps, 0.1);
	var verbosity = opt(opts.verbosity, 0);
	var recordStepStats = opt(opts.recordStepStats, false);

	// Define the inference coroutine
	var vco = {
		erpScoreRaw: function(erp, params, val) {
			this.score += erp.score(params, val);
		},
		erpScoreAD: function(erp, params, val) {
			this.score = ad_add(this.score, erp.adscore(params, val));
		},
		sample: function(erp, params) {
			var val = this.choices[this.choiceIndex];
			if (val === undefined) {
				// We don't store tapes in the trace, just raw numbers, so that
				//    re-running with the target program works correctly.
				var pparams = params.map(function(x) { return primal(x); });
				val = erp.sample(pparams);
				this.choices.push(val);
			}
			this.erpScore(erp, params, val);
			this.choiceIndex++;
			return val;
		},
		factorRaw: function(num) {
			this.score += num;
		},
		factorAD: function(num) {
			this.score = ad_add(this.score, num);
		},
		run: function(thunk, ad) {
			this.choices = [];
			return this.rerun(thunk, ad);
		},
		rerun: function(thunk, ad) {
			this.paramIndex = 0;
			this.choiceIndex = 0;
			this.score = 0;
			this.erpScore = (ad ? this.erpScoreAD : this.erpScoreRaw);
			this.factor = (ad ? this.factorAD : this.factorRaw);
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

	// Prep stats, if requeste
	var stepStats = null;
	if (recordStepStats) {
		stepStats = {
			time: [],
			elbo: []
		};
	}

	var tStart = present();

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
		var sumScoreDiff = 0.0;
		for (var s = 0; s < nSamples; s++) {
			if (verbosity > 3)
				console.log('  Sample ' + s + '/' + nSamples);
			var grad = vco.run(guideGradThunk, true);
			var guideScore = primal(vco.score);
			vco.rerun(targetThunk);
			var targetScore = vco.score;
			var scoreDiff = targetScore - guideScore;
			sumScoreDiff += scoreDiff;
			var weightedGrad = numeric.mul(grad, scoreDiff);
			if (verbosity > 4) {
				console.log('    guide score: ' + guideScore + ', target score: ' + targetScore
					+ ', diff: ' + scoreDiff);
				console.log('    grad: ' + grad.toString());
				console.log('    weightedGrad: ' + weightedGrad.toString());
			}
			numeric.addeq(sumGrad, grad);
			numeric.addeq(sumWeightedGrad, weightedGrad);
			numeric.poweq(grad, 2);	// grad is now gradSq
			numeric.addeq(sumGradSq, grad);
			var weightedGradSq = numeric.mul(grad, scoreDiff)
			numeric.addeq(sumWeightedGradSq, weightedGradSq);
		}
		var elboEst = sumScoreDiff / nSamples;
		// Record some statistics, if requested
		if (recordStepStats) {
			stepStats.time.push((present() - tStart)/1000);
			stepStats.elbo.push(elboEst);
		}
		// Compute AdaGrad learning rate and control variate,
		//    then do parameter update
		var aStar = numeric.div(sumWeightedGradSq, sumGradSq);
		numeric.muleq(aStar, sumGrad);
		var elboGradEst = numeric.sub(sumWeightedGrad, aStar);
		if (verbosity > 2) {
			console.log('  sumGrad: ' +  sumGrad.toString());
			console.log('  sumWeightedGrad: ' +  sumWeightedGrad.toString());
			console.log('  elboGradEst: ' +  elboGradEst.toString());
		}
		var maxDelta = 0;
		for (var i = 0; i < params.length; i++) {
			var grad = elboGradEst[i] / nSamples;
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
	var tEnd = present();
	if (verbosity > 0) {
		if (converged)
			console.log('CONVERGED after step ' + currStep + ' (' + maxDeltaAvg + ' < ' + convergeEps + ')');
		else
			console.log('DID NOT CONVERGE (' + maxDeltaAvg + ' > ' + convergeEps + ')');
	}

	// Restore original coroutine
	oldCoroutine = oldCoroutine;

	var ret = {
		converged: converged,
		stepsTaken: currStep,
		timeTaken: (tEnd - tStart)/1000,
		elbo: elboEst,
		params: params
	};
	if (recordStepStats) ret.stepStats = stepStats;
	return ret;
}

function sample(erp, params) {
	return coroutine.sample(erp, params);
}

function factor(num) {
	coroutine.factor(num);
}

// Create/lookup a param.
// May have an initial val, as well as an ERP.
// The ERP may be used to sample an initial val (if 'initialVal' is undefined).
// The ERP may also be used a prior score (if 'prior' is true).
function param(params, initialVal, ERP, hypers, prior) {
	if (coroutine.paramIndex == params.length) {
		if (initialVal === undefined)
			initialVal = ERP.sample(hypers);
		params.push(primal(initialVal));
	}
	var p = params[coroutine.paramIndex];
	coroutine.paramIndex++;
	if (prior)
		factor(ERP.score(hypers, p));
	return p;
}

module.exports = {
	run: run,
	infer: infer,
	sample: sample,
	factor: factor,
	param: param
};



