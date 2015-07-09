'use strict';

var numeric = require('numeric');
var present = require('present');
var assert = require('assert');
var _ = require('underscore');

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
	var regularize = opt(opts.regularize, undefined);

	// Define the regularizer for variational parameters
	if (regularize !== undefined) {
		var rweight = regularize.weight;
		if (regularize.method === 'L2') {
			regularize = function(p0, p1, learningRate) {
				return p1 - learningRate * rweight * p0;
			};
		} else
		if (regularize.method === 'L1') {
			// 'Clipped' L1 regularization for stochastic gradient descent.
			// Sources:
			// https://lingpipe.files.wordpress.com/2008/04/lazysgdregression.pdf
			// http://aclweb.org/anthology/P/P09/P09-1054.pdf
			regularize = function(p0, p1, learningRate) {
				if (p1 > 0.0) {
					return Math.max(0.0, p1 - rweight * learningRate);
				}
				else if (p1 < 0.0) {
					return Math.min(0.0, p1 + rweight * learningRate);
				}
				else return p1;
			}
		}
	} else regularize = function(p0, p1, learningRate) { return p1; };

	// Define the inference coroutine
	function Trace() {};
	Trace.prototype = {
		erpScoreRaw: function(erp, params, val) {
			var score = erp.score(params, val);
			this.score += score;
			return score;
		},
		erpScoreAD: function(erp, params, val) {
			var score = erp.adscore(params, val);
			this.score = ad_add(this.score, score);
			return score;
		},
		sample: function(erp, params) {
			var val = this.choices[this.choiceIndex];
			if (val === undefined) {
				// We don't store tapes in the trace, just raw numbers, so that
				//    re-running with the target program works correctly.
				var pparams = params.map(function(x) { return primal(x); });
				val = erp.sample(pparams);
				this.choices.push(val);
				this.choiceInfo.push({ erp: erp, params: params, score: 0.0 });
			}
			var score = this.erpScore(erp, params, val);
			this.choiceInfo[this.choiceIndex].score = score;
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
			this.choiceInfo = [];
			return this.rerun(thunk, ad);
		},
		rerun: function(thunk, ad) {
			this.paramIndex = 0;
			this.choiceIndex = 0;
			this.score = 0;
			this.erpScore = (ad ? this.erpScoreAD : this.erpScoreRaw);
			this.factor = (ad ? this.factorAD : this.factorRaw);
			return thunk();
		},
		mhstep: function(thunk, ad) {
			// Make proposal
			var i = Math.floor(Math.random()*this.choices.length);
			var info = this.choiceInfo[i];
			var currval = this.choices[i];
			var rvsLP = info.score;
			var newval = info.erp.sample(info.params);
			var fwdLP = info.erp.score(params, newval);
			this.choices[i] = newval;

			// Run trace update
			var oldChoices = _.clone(this.choices);
			var oldChoiceInfo = _.clone(this.choiceInfo);
			var oldScore = this.score;
			var ret = this.rerun(thunk, ad);

			// Accept/reject
			var acceptThresh = Math.min(1.0, Math.exp(this.score - oldScore + rvsLP - fwdLP));
			if (Math.random() < acceptThresh) {
				this.choices = oldChoices;
				this.choiceInfo = oldChoiceInfo;
				this.score = oldScore;
			}

			return ret;
		}
	}

	// Install coroutine
	var oldCoroutine = coroutine;
	coroutine = new Trace();

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
		return coroutine.score;
	})
	var guideGradThunk = function() {
		return guideGrad(params);
	}

	// Prep step stats, if requested
	var stepStats = null;
	if (recordStepStats) {
		stepStats = {
			time: [],
			elbo: []
		};
	}

	// Estimate the parameter gradient using the ELBO
	var estimateGradientELBO = function(useEmpiricalMeans, componentWiseAStar) {
		// Initialize accumulators
		var sumGrad = numeric.rep([params.length], 0.0);
		var sumWeightedGrad = numeric.rep([params.length], 0.0);
		var sumScoreDiff = 0.0;
		if (!useEmpiricalMeans) {
			var sumWeightedGradSq = numeric.rep([params.length], 0.0);
			var sumGradSq = numeric.rep([params.length], 0.0);
		} else {
			var gradSamps = [];
			var weightedGradSamps = [];
		}
		// Draw samples from the guide, score using the target
		for (var s = 0; s < nSamples; s++) {
			if (verbosity > 3)
				console.log('  Sample ' + s + '/' + nSamples);
			var grad = coroutine.run(guideGradThunk, true);
			var guideScore = primal(coroutine.score);
			coroutine.rerun(targetThunk);
			var targetScore = coroutine.score;
			var scoreDiff = targetScore - guideScore;
			sumScoreDiff += scoreDiff;
			var weightedGrad = numeric.mul(grad, scoreDiff);
			if (verbosity > 4) {
				console.log('    guide score: ' + guideScore + ', target score: ' + targetScore
					+ ', diff: ' + scoreDiff);
				console.log('    grad: ' + grad.toString());
				console.log('    weightedGrad: ' + weightedGrad.toString());
			}
			assert(isFinite(scoreDiff),
				'Detected non-finite score(s)! ERP params have probably moved outside their support...');
			numeric.addeq(sumGrad, grad);
			numeric.addeq(sumWeightedGrad, weightedGrad);
			if (!useEmpiricalMeans) {
				numeric.poweq(grad, 2);	// grad is now gradSq
				numeric.addeq(sumGradSq, grad);
				var weightedGradSq = numeric.mul(grad, scoreDiff)
				numeric.addeq(sumWeightedGradSq, weightedGradSq);
			} else {
				gradSamps.push(grad);
				weightedGradSamps.push(weightedGrad);
			}
		}
		if (!useEmpiricalMeans) {
			var covarVec = sumWeightedGradSq;
			var varVec = sumGradSq;
		} else {
			var gradMean = numeric.div(sumGrad, nSamples);
			var weightedGradMean = numeric.div(sumWeightedGrad, nSamples);
			var gradVar = numeric.rep([params.length], 0.0);
			var covar = numeric.rep([params.length], 0.0);
			for (var s = 0; s < nSamples; s++) {
				var normGrad = numeric.sub(gradSamps[s], gradMean);
				numeric.addeq(gradVar, numeric.mul(normGrad, normGrad));
				var normWeightedGrad = numeric.sub(weightedGradSamps[s], weightedGradMean);
				numeric.addeq(covar, numeric.mul(normGrad, normWeightedGrad));
			}
			// Not sure about these...
			numeric.diveq(gradVar, nSamples);
			numeric.diveq(covar, nSamples);
			var covarVec = covar;
			var varVec = gradVar;
		}
		var elboEst = sumScoreDiff / nSamples;
		// Compute AdaGrad learning rate and control variate,
		//    then do parameter update
		if (componentWiseAStar) {
			var aStar = numeric.div(covarVec, varVec);
			numeric.muleq(aStar, sumGrad);
			var elboGradEst = numeric.sub(sumWeightedGrad, aStar);
		} else {
			var numerSum = numeric.sum(covarVec);
			var denomSum = numeric.sum(varVec);
			var aStar = numerSum / denomSum;
			var offset = numeric.mul(sumGrad, aStar);
			var elboGradEst = numeric.sub(sumWeightedGrad, offset);
		}
		numeric.muleq(elboGradEst, 1.0/nSamples);
		if (verbosity > 2) {
			console.log('  sumGrad: ' +  sumGrad.toString());
			console.log('  sumWeightedGrad: ' +  sumWeightedGrad.toString());
			console.log('  elboGradEst: ' +  elboGradEst.toString());
		}
		return {
			grad: elboGradEst,
			elbo: elboEst
		};
	}

	// Estimate the parameter gradient using the EUBO
	var estimateGradientEUBO = function() {
		// TODO: Fill this in
	}

	var tStart = present();

	// Run guide once to initialize vector of params
	// TODO: This will not work if params has variable size--we'll need to
	//    adopt a different strategy then.
	coroutine.run(guideThunk);
	// Do variational inference
	var currStep = 0;
	var maxDeltaAvg = 0.0;
	var runningG2 = numeric.rep([params.length], 0.0);
	do {
		if (verbosity > 1)
			console.log('Variational iteration ' + (currStep+1) + '/' + nSteps);
		if (verbosity > 2)
			console.log('  params: ' + params.toString());
		var est = estimateGradientELBO(false, false);
		var gradEst = est.grad;
		var elboEst = est.elbo;
		// Record some statistics, if requested
		if (recordStepStats) {
			stepStats.time.push((present() - tStart)/1000);
			stepStats.elbo.push(elboEst);
		}
		var maxDelta = 0;
		for (var i = 0; i < params.length; i++) {
			var grad = gradEst[i];
			runningG2[i] += grad*grad;
			var weight = learnRate / Math.sqrt(runningG2[i]);
			assert(isFinite(weight),
				'Detected non-finite AdaGrad weight! There are probably zeroes in the gradient...');
			var delta = weight * grad;
			var p0 = params[i];
			var p1 = p0 + delta;
			params[i] = regularize(p0, p1, weight);
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
// May have an initial val, as well as a random sampler.
// The sampler may be used to sample an initial val (if 'initialVal' is undefined).
function param(params, initialVal, sampler, hypers) {
	if (coroutine.paramIndex == params.length) {
		if (initialVal === undefined)
			initialVal = sampler(hypers);
		params.push(primal(initialVal));
	}
	var p = params[coroutine.paramIndex];
	coroutine.paramIndex++;
	return p;
}

module.exports = {
	run: run,
	infer: infer,
	sample: sample,
	factor: factor,
	param: param
};



