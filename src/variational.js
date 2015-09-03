'use strict';

var numeric = require('numeric');
var present = require('present');
var assert = require('assert');
var _ = require('underscore');
var fs = require('fs');



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
function Trace() {};
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
		this.score += num;
		if (!isFinite(this.score)) {
			console.log('old score: ' + oldscore);
			console.log('factor score: ' + num);
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
	var getGrad = function(trace, propName, alsoGetVals) {
		var traceProp = trace[propName];
		traceProp.determineFanout();
		// traceProp.reversePhaseDebug(1.0);
      	traceProp.reversePhase(1.0);
      	// traceProp.print();
      	var ret = { gradient: {} };
      	if (alsoGetVals) ret.vals = {};
      	for (var name in params.values) {
      		var p = params.values[name];
      		if (alsoGetVals) ret.vals[name] = p.primal;
      		if (params.used[name]) {
      			if (!allowZeroDerivatives && p.sensitivity === 0.0) {
      				// traceProp.print();
      				// console.log('-------------------------------------------------');
      				console.log('name: ' + name);
      				console.log('id: ' + p.id);
      				assert(false, 'Found zero in guide ' + propName + ' gradient!');
      			}
      			ret.gradient[name] = p.sensitivity;
      		}
      	}
      	return ret;
	}
	return function(trace) {
		params.values = objMap(params.values, function(p) {
			return ad_maketape(p);
		});
		params.used = {};
		guide('', params, args);
		var scoreRet = getGrad(trace, 'score', true);
		var grads = {
			scoreGrad: scoreRet.gradient
		};
		if (trace.entropy !== undefined) {
			trace.score.resetState();
			grads.entropyGrad = getGrad(trace, 'entropy').gradient;
		}
		params.values = scoreRet.vals;
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
	} else throw 'Unrecognized optimizeMethod "' + optimizeMethod + '"';

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
	var estimateGradientELBO = function(componentWiseAStar) {
		// Initialize accumulators
		var sumScoreDiff = 0.0;
		var sumGrad = {};
		var sumWeightedGrad = {};
		var sumGradSq = {};
		var sumWeightedGradSq = {};
		// Draw samples from the guide, score using the target
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
				if (!sumGrad.hasOwnProperty(name)) {
					sumGrad[name] = 0.0;
					sumWeightedGrad[name] = 0.0;
					sumGradSq[name] = 0.0;
					sumWeightedGradSq[name] = 0.0;
				}
				sumGrad[name] += g;
				var weightedGrad = g * scoreDiff;
				if (doEntropyRegularization)
					weightedGrad += entropyRegularizeWeight * grads.entropyGrad[name];
				sumWeightedGrad[name] += weightedGrad;
				var gSq = g*g;
				sumGradSq[name] += gSq;
				var weightedGradSq = gSq * scoreDiff;
				sumWeightedGradSq[name] += weightedGradSq;
			}
		}
		// Compute AdaGrad learning rate and control variate
		var elboGradEst = {};
		if (componentWiseAStar) {
			for (var name in sumGrad) {
				var aStar = sumWeightedGradSq[name] / sumGradSq[name];
				elboGradEst[name] = (sumWeightedGrad[name] - sumGrad[name]*aStar)/nSamples;
			}
		} else {
			var numerSum = 0.0;
			var denomSum = 0.0;
			for (var name in sumGrad) {
				numerSum += sumWeightedGradSq[name];
				denomSum += sumGradSq[name];
			}
			var aStar = numerSum / denomSum;
			for (var name in sumGrad) {
				elboGradEst[name] = (sumWeightedGrad[name] - sumGrad[name]*aStar)/nSamples;
				if (!allowZeroDerivatives && elboGradEst[name] === 0.0) {
					console.log('name: ' + name);
					console.log('sumWeightedGrad: ' + sumWeightedGrad[name]);
					console.log('sumGrad*aStar: ' + sumGrad[name]*aStar);
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
	var estimateGradientEUBO = function() {
		var gradEst = {};
		var sumScoreDiff = 0.0;
		for (var s = 0; s < nSamples; s++) {
			if (verbosity > 3)
				console.log('  Sample ' + s + '/' + nSamples);
			var chain = getPosteriorSample();
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
				if (!gradEst.hasOwnProperty(name))
					gradEst[name] = 0.0;
				gradEst[name] += g;
			}
		}
		for (var name in gradEst)
			gradEst[name] /= nSamples;
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
		var estimateGradient = function() { return estimateGradientELBO(false); };
	} else if (gradientOpts.method === 'EUBO') {
		var estimateGradient = estimateGradientEUBO;
	} else if (gradientOpts.method == 'ELBO|EUBO') {
		var mixWeight = gradientOpts.mixWeight;
		var estimateGradient = function() {
			if (Math.random() < mixWeight)
				return estimateGradientELBO(false);
			else
				return estimateGradientEUBO();
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
			if (!allowZeroDerivatives && grad === 0.0) {
				console.log('name: ' + name);
				console.log('grad: ' + grad);
				assert(false,
				'Detected a zero in the gradient!');
			}
			var weight;
			if (optimizeMethod === 'AdaGrad') {
				if (!runningG2.hasOwnProperty(name))
					runningG2[name] = 0.0;
				runningG2[name] += grad*grad;
				weight = learnRate / Math.sqrt(runningG2[name]);
			} else if (optimizeMethod === 'Adam') {
				if (!runningM.hasOwnProperty(name)) {
					runningM[name] = 0;
					runningV[name] = 0;
				}
				runningM[name] = mDecayRate*runningM[name] + (1-mDecayRate)*grad;
				runningV[name] = vDecayRate*runningV[name] + (1-vDecayRate)*(grad*grad);
				var mt = runningM[name] / (1 - Math.pow(mDecayRate, currStep+1));
				var vt = runningV[name] / (1 - Math.pow(vDecayRate, currStep+1));
				weight = stepSize * mt / (Math.sqrt(vt) + adamEps);
			}
			if (!isFinite(weight)) {
				console.log('name: ' + name);
				console.log('grad: ' + grad);
				if (optimizeMethod === 'AdaGrad')
					console.log('runningG2: ' + runningG2[name]);
				else if (optimizeMethod === 'Adam') {
					console.log('runningM: ' + runningM[name]);
					console.log('runningV ' + runningV[name]);
				}
				assert(false, "Detected non-finite param update weight!");
			}
			var delta = weight * grad;
			var p0 = params.values[name];
			var p1 = p0 + delta;
			params.values[name] = regularize(p0, p1, weight);
			// When recording changes, do this in the transformed space
			var t = params.transforms[name];
			if (t !== undefined) {
				var tp1 = t.fwd(p1);
				var tp0 = t.fwd(p0);
				delta = tp1 - tp0;
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
}

function sample(name, erp, params) {
	return coroutine.sample(name, erp, params);
}

function factor(name, num) {
	coroutine.factor(name, num);
}

// Create/lookup a param.
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
		params.values[name] = ad_maketape(ad_primal(initialVal));
	}
	params.transforms[name] = transform;
	params.used[name] = true;
	var p = params.values[name];
	if (transform !== undefined)
		return transform.fwd(p);
	else
		return p;
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

module.exports = {
	variational: {
		makeParams: makeParams,
		infer: infer,
		saveParams: saveParams,
		loadParams: loadParams
	},
	sample: sample,
	factor: factor,
	param: param
};



