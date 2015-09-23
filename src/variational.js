'use strict';

var numeric = require('numeric');
var present = require('present');
var assert = require('assert');
var _ = require('underscore');
var fs = require('fs');
var tensor = require('./tensor');



// Global inference coroutine
var coroutine = {
	sample: function(name, erp, params) {
		var pparams = params.map(function(x) { return ad_primal(x); });
		return erp.sample(pparams);
	},
	factor: function(name, num) {}
};


function makeParams() {
	return {
		values: {},
		transforms: {},
		used: {}
	};
}


// Define the variational inference coroutine
function Trace() { this.temp = 1; };
Trace.prototype = {
	copy: function(otherTrace) {
		this.score = otherTrace.score;
		this.numChoices = otherTrace.numChoices;
		this.choices = [];
		for (var name in otherTrace.choices) {
			var c = otherTrace.choices[name];
			this.choices[name] = {
				val: c.val, erp: c.erp, params: c.params, score: c.score, reachable: true
			};
		}
		this.returnVal = otherTrace.returnVal;
		if (this.entropy !== undefined) {
			this.entropy = otherTrace.entropy;
		}
	},
	erpScoreRaw: function(name, erp, params, val, newchoice) {
		var score = erp.score(params, val);
		var oldscore = this.score;
		this.score += score;
		if (!isFinite(score)) {
			console.log('name: ' + name);
			console.log('val: ' + val);
			console.log('params: ' + params);
			console.log('scorer: ' + erp.score.name);
			console.log('oldscore: ' + oldscore);
			console.log('ERP score: ' +  score);
			console.log('new score: ' + this.score);
			assert(false, 'ERP has non-finite score in target!');
		}
		if (newchoice) this.newlp += score;
		return score;
	},
	erpScoreAD: function(name, erp, params, val, newchoice) {
		var score = erp.adscore(params, val);
		var oldscore = this.score;
		this.score = ad_add(this.score, score);
		if (!isFinite(ad_primal(score))) {
			console.log('name: ' + name);
			console.log('val: ' + val);
			console.log('params: ' + params.map(function(x) {return ad_primal(x);}));
			console.log('scorer: ' + erp.score.name);
			console.log('oldscore: ' + ad_primal(oldscore));
			console.log('ERP score: ' +  ad_primal(score));
			console.log('new score: ' + ad_primal(this.score));
			assert(false, 'ERP has non-finite score in guide!');
		}
		if (newchoice) this.newlp = ad_add(this.newlp, score);
		if (this.entropy !== undefined)
			this.entropy = ad_add(this.entropy, erp.adentropy(params));
		return score;
	},
	sample: function(name, erp, params) {
		var c = this.choices[name];
		if (c === undefined) {
			// We don't store tapes in the trace, just raw numbers, so that
			//    re-running with the target program works correctly.
			var pparams = params.map(function(x) { return ad_primal(x); });
			var val = erp.sample(pparams);
			var score = this.erpScore(name, erp, params, val, true);
			c = { val: val, erp: erp, params: params, score: score, reachable: true };
			this.choices[name] = c;
			this.numChoices++;
		} else {
			c.score = this.erpScore(name, erp, params, c.val, false);
			c.reachable = true;
		}
		return c.val;
	},
	factorRaw: function(name, num) {
		var oldscore = this.score;
		this.score += this.temp * num;
		if (!isFinite(this.score)) {
			console.log('old score: ' + oldscore);
			console.log('factor score: ' + num);
			console.log('temp: ' + this.temp);
			console.log('new score: ' + this.score);
			assert(false, 'Factor has non-finite score!');
		}
	},
	factorAD: function(name, num) {
		// this.score = ad_add(this.score, num);
		throw 'Guide programs should not have factors!';
	},
	run: function(thunk, ad, entropy) {
		this.choices = {};
		this.numChoices = 0;
		return this.rerun(thunk, ad, entropy);
	},
	rerun: function(thunk, ad, entropy) {
		this.score = 0.0;
		if (entropy)
			this.entropy = 0.0;
		this.newlp = 0.0;
		this.erpScore = (ad ? this.erpScoreAD : this.erpScoreRaw);
		this.factor = (ad ? this.factorAD : this.factorRaw);
		var oldCoroutine = coroutine;
		coroutine = this;
		this.returnVal = thunk(this);
		coroutine = oldCoroutine
		return this.returnVal;
	},
	mhstep: function(thunk, ad) {
		// Make proposal
		var cindex = Math.floor(Math.random()*this.choices.length);
		var c;
		var i = 0;
		// (Could speed this up by keeping track of linear choice list)
		for (var name in this.choices) {
			c = this.choices[name];
			if (i === cindex) break;
			i++;
		}
		var rvsLP = c.score;
		var newval = c.erp.sample(c.params);
		var fwdLP = c.erp.score(c.params, newval);

		// Run trace update
		var oldTrace = new Trace(); oldTrace.copy(this);
		c.val = newval;
		for (var name in this.choices)
			this.choices[name].reachable = false;
		this.rerun(thunk, ad);

		// Clear out old choices
		var oldlp = 0.0;
		for (var name in this.choices) {
			var c = this.choices[name];
			if (!c.reachable) {
				this.numChoices--;
				oldlp += c.score;
				delete this.choices[name];
			}
		}

		// Accept/reject
		fwdLP += this.newlp - Math.log(oldTrace.numChoices);
		rvsLP += oldlp - Math.log(this.numChoices);
		var acceptProb = Math.min(1.0, Math.exp(this.score - oldTrace.score + rvsLP - fwdLP));
		// console.log('-----------------');
		// console.log('newScore: ' + this.score + ', oldScore: ' + oldTrace.score);
		// console.log('fwdLP: ' + fwdLP + ', rvsLP: ' + rvsLP);
		// console.log('acceptProb: ', acceptProb);
		if (Math.random() < acceptProb) {
			// console.log('ACCEPT');
		} else {
			// console.log('REJECT');
			this.copy(oldTrace);
		}
	}
};


// Thunks that we'll feed to 'run' and 'rerun'
function makeTargetThunk(target, args) {
	return function() {
		return target('', args);
	};
}
function makeGuideThunk(guide, params, args) {
	return function() {
		return guide('', params, args);
	};
}
function makeGuideGradThunk(guide, params, args, allowZeroDerivatives) {
	var objMap = function(obj, f) {
	  var newobj = {};
	  for (var prop in obj)
	    newobj[prop] = f(obj[prop]);
	  return newobj;
	}
	var getGrad = function(trace, propName, outParamVals) {
		var traceProp = trace[propName];
		traceProp.determineFanout();
		// traceProp.reversePhaseDebug(1.0);
      	traceProp.reversePhase(1.0);
      	// traceProp.print();
      	var ret = { gradient: {} };
      	for (var name in params.values) {
      		var p = params.values[name];
      		if (params.used[name]) {
      			// Record any newly-created parameter values
	      		if (outParamVals !== undefined && !outParamVals.hasOwnProperty(name))
	      			outParamVals[name] = tensor.map(p, function(x) { return x.primal; });
      			if (!allowZeroDerivatives && tensor.any(p, function(x) { return x.sensitivity === 0.0; })) {
      				// traceProp.print();
      				// console.log('-------------------------------------------------');
      				console.log('name: ' + name);
      				assert(false, 'Found zero in guide ' + propName + ' gradient!');
      			}
      			if (tensor.any(p, function(x) { return isNaN(x.sensitivity); })) {
      				// traceProp.print();
      				// console.log('-------------------------------------------------');
      				console.log('name: ' + name);
      				assert(false, 'Found NaN in guide ' + propName + ' gradient!');
      			}
      			ret.gradient[name] = tensor.map(p, function(x) { return x.sensitivity; });
      		}
      	}
      	return ret;
	}
	return function(trace) {
		var vals = params.values;
		params.values = objMap(params.values, function(p) {
			return tensor.map(p, function(x) { return ad_maketape(x); });
		});
		params.used = {};
		guide('', params, args);
		var scoreRet = getGrad(trace, 'score', vals);
		var grads = {
			scoreGrad: scoreRet.gradient
		};
		if (trace.entropy !== undefined) {
			trace.score.resetState();
			grads.entropyGrad = getGrad(trace, 'entropy').gradient;
		}
		params.values = vals;
      	return grads;
	}
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
	var convergeEps = opt(opts.convergeEps, 0.1);
	var verbosity = opt(opts.verbosity, 0);
	var recordStepStats = opt(opts.recordStepStats, false);
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
	var allowZeroDerivatives = opt(opts.allowZeroDerivatives, false);
	var entropyRegularizeWeight = opt(opts.entropyRegularizeWeight, 0);
	var doEntropyRegularization = (entropyRegularizeWeight !== 0);

	var optimizeMethod = opt(opts.optimizeMethod, 'AdaGrad');
	if (optimizeMethod === 'AdaGrad')
		var learnRate = opt(opts.initLearnRate, 0.5);
	else if (optimizeMethod === 'Adam') {
		var stepSize = opt(opts.stepSize, 0.001);
		var mDecayRate = opt(opts.mDecayRate, 0.9);
		var vDecayRate = opt(opts.vDecayRate, 0.99);
		var adamEps = opt(opts.adamEps, 1e-8)
	} else if (optimizeMethod === 'None') {
		var learnRate = opt(opts.initLearnRate, 0.5);
	} else {
		throw 'Unrecognized optimizeMethod "' + optimizeMethod + '"';
	}

	var tempSchedule = opt(opts.tempSchedule, function() { return 1; });


	// Regularization stuff
	var regularizationWeight = opt(opts.regularizationWeight, 0);
	// var regularize = opt(opts.regularize, undefined);
	// if (regularize !== undefined) {
	// 	var rweight = regularize.weight;
	// 	if (regularize.method === 'L2') {
	// 		regularize = function(p0, p1, learningRate) {
	// 			// return numeric.sub(p1, numeric.muleq(numeric.mul(learningRate, rweight), p0));
	// 			var ret = numeric.sub(p1, numeric.muleq(numeric.mul(learningRate, rweight), p0));
	// 			// console.log('-------------------------')
	// 			// console.log('p0:');
	// 			// console.log(p0);
	// 			// console.log('p1:');
	// 			// console.log(p1);
	// 			// console.log('learningRate:');
	// 			// console.log(learningRate);
	// 			// console.log('ret:');
	// 			// console.log(ret);
	// 			return ret;
	// 		};
	// 	} else
	// 	if (regularize.method === 'L1') {
	// 		// 'Clipped' L1 regularization for stochastic gradient descent.
	// 		// Sources:
	// 		// https://lingpipe.files.wordpress.com/2008/04/lazysgdregression.pdf
	// 		// http://aclweb.org/anthology/P/P09/P09-1054.pdf
	// 		regularize = function(p0, p1, learningRate) {
	// 			var w = numeric.mul(rweight, learningRate);
	// 			return tensor.map2(p1, w, function(p1_, w_) {
	// 				if (p1_ > 0)
	// 					return Math.max(0, p1_ - w_);
	// 				else if (p1_ < 0)
	// 					return Math.min(0, p1_ + w_);
	// 				else return p1_;
	// 			});
	// 		}
	// 	}
	// } else regularize = function(p0, p1, learningRate) { return p1; };

	// Default trace
	var trace = new Trace();

	var params = makeParams();

	var targetThunk = makeTargetThunk(target, args);
	var guideThunk = makeGuideThunk(guide, params, args);
	var guideGradThunk = makeGuideGradThunk(guide, params, args, allowZeroDerivatives);

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
	var estimateGradientELBO = function(stepNum, componentWiseAStar) {
		// Initialize accumulators
		var sumScoreDiff = 0.0;
		var sumGrad = {};
		var sumWeightedGrad = {};
		var sumGradSq = {};
		var sumWeightedGradSq = {};
		// Draw samples from the guide, score using the target
		trace.temp = tempSchedule(stepNum, nSteps);
		for (var s = 0; s < nSamples; s++) {
			if (verbosity > 3)
				console.log('  Sample ' + s + '/' + nSamples);
			var grads = trace.run(guideGradThunk, true, doEntropyRegularization);
			var scoreGrad = grads.scoreGrad; 
			var guideScore = ad_primal(trace.score);
			trace.rerun(targetThunk);
			var targetScore = trace.score;
			var scoreDiff = targetScore - guideScore;
			sumScoreDiff += scoreDiff;
			if (verbosity > 4) {
				console.log('    guide score: ' + guideScore + ', target score: ' + targetScore
					+ ', diff: ' + scoreDiff);
				console.log('    scoreGrad: ' + JSON.stringify(scoreGrad));
			}
			for (var name in scoreGrad) {
				var g = scoreGrad[name];
				var dim = numeric.dim(g);
				if (!sumGrad.hasOwnProperty(name)) {
					sumGrad[name] = numeric.rep(dim, 0);
					sumWeightedGrad[name] = numeric.rep(dim, 0);
					sumGradSq[name] = numeric.rep(dim, 0);
					sumWeightedGradSq[name] = numeric.rep(dim, 0);
				}
				numeric.addeq(sumGrad[name], g);
				var weightedGrad = numeric.mul(g, scoreDiff);
				if (doEntropyRegularization)
					numeric.addeq(weightedGrad, numeric.mul(entropyRegularizeWeight, grads.entropyGrad[name]));
				numeric.addeq(sumWeightedGrad[name], weightedGrad);
				var gSq = numeric.mul(g, g);
				numeric.addeq(sumGradSq[name], gSq);
				var weightedGradSq = numeric.mul(gSq, scoreDiff);
				numeric.addeq(sumWeightedGradSq[name], weightedGradSq);
			}
		}
		// Compute AdaGrad learning rate and control variate
		var elboGradEst = {};
		if (componentWiseAStar) {
			for (var name in sumGrad) {
				var aStar = numeric.div(sumWeightedGradSq[name] / sumGradSq[name]);
				elboGradEst[name] = numeric.div(numeric.sub(sumWeightedGrad[name], numeric.mul(sumGrad[name], aStar)), nSamples);
			}
		} else {
			var numerSum = 0.0;
			var denomSum = 0.0;
			for (var name in sumGrad) {
				numerSum += numeric.sum(sumWeightedGradSq[name]);
				denomSum += numeric.sum(sumGradSq[name]);
			}
			var aStar = numerSum / denomSum;
			for (var name in sumGrad) {
				elboGradEst[name] = numeric.div(numeric.sub(sumWeightedGrad[name], numeric.mul(sumGrad[name], aStar)), nSamples);
				if (!allowZeroDerivatives && tensor.any(elboGradEst[name], function(x) { x === 0; })) {
					console.log('name: ' + name);
					console.log('sumWeightedGrad: ' + sumWeightedGrad[name]);
					console.log('sumGrad: ' + sumGrad[name]);
					console.log('sumGrad*aStar: ' + sumGrad[name]*aStar);
					console.log('sumScoreDiff: ' + sumScoreDiff);
					assert(false, 'zero in elboGradEst! - usually only happens when nSamples = 1.');
				}
			}
		}
		if (verbosity > 3) {
			console.log('  sumGrad: ' +  JSON.stringify(sumGrad));
			console.log('  sumWeightedGrad: ' +  JSON.stringify(sumWeightedGrad));
			console.log('  elboGradEst: ' +  JSON.stringify(elboGradEst));
		}
		return {
			grad: elboGradEst,
			elbo: sumScoreDiff / nSamples
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
	var estimateGradientEUBO = function(stepNum) {
		var gradEst = {};
		var sumScoreDiff = 0.0;
		for (var s = 0; s < nSamples; s++) {
			if (verbosity > 3)
				console.log('  Sample ' + s + '/' + nSamples);
			var chain = getPosteriorSample();
			chain.temp = tempSchedule(stepNum, nSteps);
			var targetScore = chain.score;
			// Save the trace so we can restore it after the AD gradient run
			var oldChain = new Trace(); oldChain.copy(chain);
			var gradient = chain.rerun(guideGradThunk, true).scoreGrad;
			var guideScore = ad_primal(chain.score);
			chain.copy(oldChain);
			var scoreDiff = targetScore - guideScore;
			if (verbosity > 4) {
				console.log('    guide score: ' + guideScore + ', target score: ' + targetScore
					+ ', diff: ' + scoreDiff);
				console.log('    grad: ' + JSON.stringify(grad));
			}
			sumScoreDiff += scoreDiff;
			for (var name in gradient) {
				var g = gradient[name];
				var dim = numeric.dim(g);
				if (!gradEst.hasOwnProperty(name))
					gradEst[name] = numeric.rep(dim, 0);
				numeric.addeq(gradEst[name], g);
			}
		}
		for (var name in gradEst)
			numeric.diveq(gradEst[name], nSamples);
		if (verbosity > 2) {
			console.log('  gradEst: ' +  JSON.stringify(gradEst));
		}
		return {
			grad: gradEst,
			eubo: sumScoreDiff / nSamples
		}
	}

	// Pre-define the function that'll compute the gradient estimator, depending
	//    upon which method was requested
	if (gradientOpts.method === 'ELBO') {
		var estimateGradient = function(stepNum) { return estimateGradientELBO(stepNum, false); };
	} else if (gradientOpts.method === 'EUBO') {
		var estimateGradient = estimateGradientEUBO;
	} else if (gradientOpts.method == 'ELBO|EUBO') {
		var mixWeight = gradientOpts.mixWeight;
		var estimateGradient = function(stepNum) {
			if (Math.random() < mixWeight)
				return estimateGradientELBO(stepNum, false);
			else
				return estimateGradientEUBO(stepNum);
		};
	}

	var tStart = present();

	// Do variational inference
	var currStep = 0;
	var maxDeltaAvg = 0.0;
	if (optimizeMethod === 'AdaGrad')
		var runningG2 = {};
	else if (optimizeMethod === 'Adam') {
		var runningM = {};
		var runningV = {};
	}
	do {
		if (verbosity > 1)
			console.log('Variational iteration ' + (currStep+1) + '/' + nSteps);
		if (verbosity > 3)
			console.log('  params: ' + JSON.stringify(params.values));
		var est = estimateGradient(currStep);
		var gradEst = est.grad;
		// Record some statistics, if requested
		if (recordStepStats) {
			stepStats.time.push((present() - tStart)/1000);
			var elboEst = est.elbo === undefined ? estimateELBO() : est.elbo;
			stepStats.elbo.push(elboEst);
			var euboEst = est.eubo === undefined ? estimateEUBO() : est.eubo;
			stepStats.eubo.push(euboEst);
		}
		if (verbosity > 2) {
			var str = '  ';
			if (est.elbo !== undefined) {
				str += 'elbo : ' + est.elbo;
				if (est.eubo !== undefined)
					str += ', ';
			}
			if (est.eubo !== undefined) {
				str += 'eubo: ' + est.eubo;
			}
			console.log(str);
		}
		var maxDelta = 0;
		for (var name in gradEst) {
			var grad = gradEst[name];
			if (regularizationWeight > 0)
				numeric.subeq(grad, numeric.mul(regularizationWeight, params.values[name]));
			var dim = numeric.dim(grad);
			if (!allowZeroDerivatives && tensor.any(grad, function(x) { x === 0.0; })) {
				console.log('name: ' + name);
				console.log('grad: ' + grad);
				assert(false,
				'Detected a zero in the gradient!');
			}
			var weight;
			if (optimizeMethod === 'None') {
				weight = learnRate;
			} else if (optimizeMethod === 'AdaGrad') {
				if (!runningG2.hasOwnProperty(name))
					runningG2[name] = numeric.rep(dim, 0);
				numeric.addeq(runningG2[name], numeric.mul(grad, grad));
				weight = numeric.div(learnRate, numeric.sqrt(runningG2[name]));
			} else if (optimizeMethod === 'Adam') {
				if (!runningM.hasOwnProperty(name)) {
					runningM[name] = numeric.rep(dim, 0);
					runningV[name] = numeric.rep(dim, 0);
				}
				runningM[name] = numeric.add(numeric.mul(mDecayRate, runningM[name]), numeric.mul(1-mDecayRate), grad);
				runningV[name] = numeric.add(numeric.mul(vDecayRate, runningV[name]), numeric.mul(1-vDecayRate), numeric.mul(grad, grad));
				var mt = numeric.div(runningM[name], (1 - Math.pow(mDecayRate, currStep+1)));
				var vt = numeric.div(runningV[name], (1 - Math.pow(vDecayRate, currStep+1)));
				weight = numeric.div(numeric.mul(stepSize, mt), numeric.add(numeric.sqrt(vt), adamEps));
			}
			if (!numeric.all(numeric.isFinite(weight))) {
				console.log('name: ' + name);
				console.log('grad: ' + grad);
				console.log('weight: ' + weight);
				if (optimizeMethod === 'AdaGrad')
					console.log('runningG2: ' + runningG2[name]);
				else if (optimizeMethod === 'Adam') {
					console.log('runningM: ' + runningM[name]);
					console.log('runningV ' + runningV[name]);
				}
				assert(false, "Detected non-finite param update weight!");
			}

			var delta = numeric.mul(weight, grad);
			numeric.addeq(params.values[name], delta);
			// var p0 = params.values[name];
			// var p1 = numeric.add(p0, delta);
			// params.values[name] = regularize(p0, p1, weight);

			// When recording changes to scalar params, do this in the transformed space
			var t = params.transforms[name];
			if (t !== undefined) {
				var tp1 = t.fwd(p1[0]);
				var tp0 = t.fwd(p0[0]);
				delta = [tp1 - tp0];
			}
			maxDelta = Math.max(tensor.maxreduce(numeric.abs(delta)), maxDelta);
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
}

function sample(name, erp, params) {
	return coroutine.sample(name, erp, params);
}

function factor(name, num) {
	coroutine.factor(name, num);
}

// Create/lookup a scalar param.
// May have an initial val, a transform, and  a random sampler.
// Transform specifies how the value should be transformed.
// The sampler may be used to sample an initial val (if 'initialVal' is undefined).
function param(name, params, initialVal, transform, sampler, hypers, outnames) {
	// If caller needs to know the address of this param
	if (outnames !== undefined)
		outnames.push(name);
	if (!params.values.hasOwnProperty(name)) {
		if (initialVal === undefined)
			initialVal = sampler(hypers);
		if (transform !== undefined) {
			var origVal = initialVal;
			initialVal = transform.rvs(initialVal);
			if (!isFinite(initialVal)) {
				console.log('name: ' + name);
				console.log('raw initial val: ' + origVal);
				console.log('transformed initial val: ' + initialVal);
				assert(false, 'initial parameter value is non-finite!');
			}
		}
		params.values[name] = [ad_maketape(ad_primal(initialVal))];
	}
	params.transforms[name] = transform;
	params.used[name] = true;
	var p = params.values[name][0];
	if (transform !== undefined)
		return transform.fwd(p);
	else
		return p;
}

// Create/lookup a matrix/vector param
function paramTensor(name, params, dim, initialVal, sampler, hypers) {
	if (!params.values.hasOwnProperty(name)) {
		var val;
		if (initialVal !== undefined) {
			val = numeric.rep(dim, initialVal);
		} else {
			val = tensor.create(dim, function() { return sampler(hypers); });
		}
		tensor.mapeq(val, function(x) { return ad_maketape(x); });
		params.values[name] = val;
	}
	params.used[name] = true;
	return params.values[name];
}

// IO for parameters
function saveParams(params, filename) {
	fs.writeFileSync(filename, JSON.stringify(params.values));
};
function loadParams(filename) {
	var params = makeParams();
	params.values = JSON.parse(fs.readFileSync(filename).toString());
	return params;
};
function getTransformedParamVals(params) {
	var prms = {};
	for (var name in params.values) {
		var val = params.values[name];
		var t = params.transforms[name];
		if (t !== undefined) {
			val = t.fwd(val);
		}
		prms[name] = val;
	}
	return prms;
}

module.exports = {
	variational: {
		makeParams: makeParams,
		infer: infer,
		saveParams: saveParams,
		loadParams: loadParams,
		getTransformedParamVals: getTransformedParamVals
	},
	sample: sample,
	factor: factor,
	param: param,
	paramTensor: paramTensor
};



