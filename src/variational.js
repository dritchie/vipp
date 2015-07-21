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
	var gradientOpts = opt(opts.gradientOpts, {
		method: 'ELBO'
	});
	if (gradientOpts.method !== 'ELBO') {
		gradientOpts.nChains = opt(gradientOpts.nChains, 1);
		gradientOpts.burnIn = opt(gradientOpts.burnIn, 1000);
		gradientOpts.lag = opt(gradientOpts.lag, 0);
	}
	if (gradientOpts.method === 'ELBO+EUBO' || gradientOpts.method === 'ELBO|EUBO') {
		gradientOpts.mixWeight = opt(gradientOpts.mixWeight, 0.5);
	}

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
			this.choiceInfo[this.choiceIndex] = { erp: erp, params: params, score: score };
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
			var oldCoroutine = coroutine;
			coroutine = this;
			this.returnVal = thunk();
			coroutine = oldCoroutine
			return this.returnVal;
		},
		mhstep: function(thunk, ad) {
			// Make proposal
			var cindex = Math.floor(Math.random()*this.choices.length);
			var info = this.choiceInfo[cindex];
			var currval = this.choices[cindex];
			var rvsLP = info.score;
			var newval = info.erp.sample(info.params);
			var fwdLP = info.erp.score(info.params, newval);

			// Run trace update
			var oldChoices = _.clone(this.choices);
			var oldChoiceInfo = _.clone(this.choiceInfo);
			var oldScore = this.score;
			var oldRetVal = this.returnVal;
			this.choices[cindex] = newval;
			this.rerun(thunk, ad);

			// Accept/reject
			var acceptProb = Math.min(1.0, Math.exp(this.score - oldScore + rvsLP - fwdLP));
			// console.log('-----------------');
			// console.log('newScore: ' + this.score + ', oldScore: ' + oldScore);
			// console.log('fwdLP: ' + fwdLP + ', rvsLP: ' + rvsLP);
			// console.log('acceptProb: ', acceptProb);
			if (Math.random() < acceptProb) {
				// console.log('ACCEPT');
			} else {
				// console.log('REJECT');
				this.choices = oldChoices;
				this.choiceInfo = oldChoiceInfo;
				this.score = oldScore;
				this.returnVal = oldRetVal;
			}
		}
	}

	// Default trace
	var trace = new Trace();

	var params = {
		values: [],
		transforms: []
	}

	// Thunks that we'll feed to 'run' and 'rerun'
	var targetThunk = function() {
		return target(args);
	}
	var guideThunk = function() {
		return guide(params, args);
	}
	var guideGrad = ad_gradientR(function(p) {
		var oldvals = params.values;
		params.values = p;
		guide(params, args);
		params.values = oldvals;
		return coroutine.score;
	})
	var guideGradThunk = function() {
		return guideGrad(params.values);
	}

	// Prep step stats, if requested
	var stepStats = null;
	if (recordStepStats) {
		stepStats = {
			time: [],
			elbo: [],
			eubo: []
		};
	}

	// Estimate the ELBO
	var estimateELBO = function() {
		// Initialize accumulators
		var nParams = params.values.length;
		var sumScoreDiff = 0.0;
		for (var s = 0; s < nSamples; s++) {
			trace.run(guideThunk);
			var guideScore = trace.score;
			trace.rerun(targetThunk);
			var targetScore = trace.score;
			sumScoreDiff += (targetScore - guideScore);
		}
		return sumScoreDiff / nSamples;
	}

	// Estimate the parameter gradient using the ELBO
	var estimateGradientELBO = function(useEmpiricalMeans, componentWiseAStar) {
		// Initialize accumulators
		var nParams = params.values.length;
		var sumGrad = numeric.rep([nParams], 0.0);
		var sumWeightedGrad = numeric.rep([nParams], 0.0);
		var sumScoreDiff = 0.0;
		if (!useEmpiricalMeans) {
			var sumWeightedGradSq = numeric.rep([nParams], 0.0);
			var sumGradSq = numeric.rep([nParams], 0.0);
		} else {
			var gradSamps = [];
			var weightedGradSamps = [];
		}
		// Draw samples from the guide, score using the target
		for (var s = 0; s < nSamples; s++) {
			if (verbosity > 3)
				console.log('  Sample ' + s + '/' + nSamples);
			var grad = trace.run(guideGradThunk, true);
			var guideScore = primal(trace.score);
			trace.rerun(targetThunk);
			var targetScore = trace.score;
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
			var gradVar = numeric.rep([nParams], 0.0);
			var covar = numeric.rep([nParams], 0.0);
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


	var nChains = gradientOpts.nChains;
	var burnIn = gradientOpts.burnIn;
	var lag = gradientOpts.lag;
	var chains = [];
	var getPosteriorSample = function() {
		if (chains.length === 0) {
			// Initialize MH chains
			for (var i = 0; i < nChains; i++) {
				var tr = new Trace();
				tr.run(targetThunk);
				for (var j = 0; j < burnIn; j++)
					tr.mhstep(targetThunk);
				chains.push(tr);
			}
		}
		// Pick a chain at random
		var chain = chains[Math.floor(Math.random()*nChains)];
		for (var l = 0; l < lag; l++)
			chain.mhstep(targetThunk);
		return chain;
	}

	// Estimate the EUBO
	var estimateEUBO = function() {
		// Initialize accumulators
		var nParams = params.values.length;
		var sumScoreDiff = 0.0;
		for (var s = 0; s < nSamples; s++) {
			var trace = getPosteriorSample();
			var targetScore = trace.score;
			trace.rerun(guideThunk);
			var guideScore = trace.score;
			trace.score = targetScore; // Restore
			sumScoreDiff += (targetScore - guideScore);
		}
		return sumScoreDiff / nSamples;
	}

	// Estimate the parameter gradient using the EUBO
	var estimateGradientEUBO = function() {
		var nParams = params.values.length;
		var gradEst = numeric.rep([nParams], 0.0);
		var sumScoreDiff = 0.0;
		for (var s = 0; s < nSamples; s++) {
			if (verbosity > 3)
				console.log('  Sample ' + s + '/' + nSamples);
			var chain = getPosteriorSample();
			var targetScore = chain.score;
			// Save the choices + choiceInfo so we can restore them after the AD gradient run
			var choices = _.clone(chain.choices);
			var choiceInfo = _.clone(chain.choiceInfo);
			var retVal = chain.returnVal;
			var gradient = chain.rerun(guideGradThunk, true);
			var guideScore = primal(chain.score);
			chain.score = targetScore;
			chain.choices = choices;
			chain.choiceInfo = choiceInfo;
			chain.returnVal = retVal;
			var scoreDiff = targetScore - guideScore;
			if (verbosity > 4) {
				console.log('    guide score: ' + guideScore + ', target score: ' + targetScore
					+ ', diff: ' + scoreDiff);
				console.log('    grad: ' + grad.toString());
			}
			sumScoreDiff += scoreDiff;
			numeric.addeq(gradEst, gradient);
		}
		var euboEst = sumScoreDiff / nSamples;
		numeric.diveq(gradEst, nSamples);
		if (verbosity > 2) {
			console.log('  gradEst: ' +  gradEst.toString());
		}
		return {
			eubo: euboEst,
			grad: gradEst
		}
	}

	// Pre-define the function that'll compute the gradient estimator, depending
	//    upon which method was requested
	if (gradientOpts.method === 'ELBO') {
		var estimateGradient = function() { return estimateGradientELBO(false, false); };
	} else if (gradientOpts.method === 'EUBO') {
		var estimateGradient = estimateGradientEUBO;
	} else if (gradientOpts.method == 'ELBO+EUBO') {
		var mixWeight = gradientOpts.mixWeight
		var estimateGradient = function() {
			var elboEst = estimateGradientELBO(false, false);
			var euboEst = estimateGradientEUBO();
			numeric.muleq(elboEst.grad, mixWeight);
			numeric.muleq(euboEst.grad, 1.0 - mixWeight);
			var comboGrad = numeric.add(elboEst.grad, euboEst.grad);
			return {
				grad: comboGrad,
				elbo: elboEst.elbo,
				eubo: euboEst.eubo
			};
		};
	} else if (gradientOpts.method == 'ELBO|EUBO') {
		var mixWeight = gradientOpts.mixWeight;
		var estimateGradient = function() {
			if (Math.random() < mixWeight)
				return estimateGradientELBO(false, false);
			else
				return estimateGradientEUBO();
		};
	}

	var tStart = present();

	// Run guide once to initialize vector of params
	// TODO: This will not work if params has variable size--we'll need to
	//    adopt a different strategy then.
	trace.run(guideThunk);
	// Do variational inference
	var currStep = 0;
	var maxDeltaAvg = 0.0;
	var nParams = params.values.length;
	var runningG2 = numeric.rep([nParams], 0.0);
	do {
		if (verbosity > 1)
			console.log('Variational iteration ' + (currStep+1) + '/' + nSteps);
		if (verbosity > 2)
			console.log('  params: ' + params.values.toString());
		var est = estimateGradient();
		var gradEst = est.grad;
		// Record some statistics, if requested
		if (recordStepStats) {
			stepStats.time.push((present() - tStart)/1000);
			var elboEst = est.elbo === undefined ? estimateELBO() : est.elbo;
			stepStats.elbo.push(elboEst);
			var euboEst = est.eubo === undefined ? estimateEUBO() : est.eubo;
			stepStats.eubo.push(euboEst);
		}
		var maxDelta = 0;
		for (var i = 0; i < nParams; i++) {
			var grad = gradEst[i];
			runningG2[i] += grad*grad;
			var weight = learnRate / Math.sqrt(runningG2[i]);
			assert(isFinite(weight),
				'Detected non-finite AdaGrad weight! There are probably zeroes in the gradient...');
			var delta = weight * grad;
			var p0 = params.values[i];
			var p1 = p0 + delta;
			params.values[i] = regularize(p0, p1, weight);
			// When recording changes, do this in the transformed space
			if (params.transforms[i] !== undefined) {
				var t = params.transforms[i];
				delta = t(p1) - t(p0);
			}
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

	var ret = {
		converged: converged,
		stepsTaken: currStep,
		timeTaken: (tEnd - tStart)/1000,
		elbo: est.elbo,
		eubo: est.eubo,
		params: params
	};
	if (chains.length > 0) {
		ret.euboChainStates = chains.map(function(c) {
			return c.returnVal;
		});
	}
	if (recordStepStats) ret.stepStats = stepStats;
	return ret;

	// // Testing out MH
	// trace.run(targetThunk);
	// for (var i = 0; i < 1000; i++) {
	// 	trace.mhstep(targetThunk);
	// }
	// return trace.returnVal;
}

function sample(erp, params) {
	return coroutine.sample(erp, params);
}

function factor(num) {
	coroutine.factor(num);
}

// Create/lookup a param.
// May have an initial val, a transform, and  a random sampler.
// Transform specifies how the value should be transformed.
// The sampler may be used to sample an initial val (if 'initialVal' is undefined).
function param(params, initialVal, transform, sampler, hypers) {
	if (coroutine.paramIndex == params.values.length) {
		if (initialVal === undefined)
			initialVal = sampler(hypers);
		params.values.push(primal(initialVal));
		params.transforms.push(transform);
	}
	var t = params.transforms[coroutine.paramIndex];
	var p = params.values[coroutine.paramIndex];
	coroutine.paramIndex++;
	if (t !== undefined)
		return t(p);
	else
		return p;
}

module.exports = {
	run: run,
	infer: infer,
	sample: sample,
	factor: factor,
	param: param
};



