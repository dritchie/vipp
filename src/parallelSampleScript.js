// Install header + AD stuff into the global environment.
var ad = require('./ad/functions');
for (var prop in ad) {
	global[prop] = ad[prop];
}
var header = require('./header');
for (var prop in header) {
	global[prop] = header[prop];
}

// Target and guide thunks
var targetThunk;
var guideGradThunk;

// Current parameters
var params;

// Verbosity level
var verbosity;

// Define the inference coroutine
var coroutine = {
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
		factor(ERP.adscore(hypers, p));
	return p;
}


// ----------------------------------------------------------------------------


// Assumes 'source' contains top-level functions named 'target' and 'guide'
function init(source, verbosity) {
	verbosity = verbosity;
	eval(source);
	targetThunk = function() {
		return target(args);
	}
	var guideGrad = ad_gradientR(function(p) {
		guide(p, args);
		return vco.score;
	})
	guideGradThunk = function() {
		return guideGrad(params);
	}
}

// Sample n samples and return summary statistics necessary for a variational
//   inference gradient step
function sample(nSamples) {
	var sumGrad = numeric.rep([params.length], 0);
	var sumGradSq = numeric.rep([params.length], 0);
	var sumWeightedGrad = numeric.rep([params.length], 0);
	var sumWeightedGradSq = numeric.rep([params.length], 0);
	var sumScoreDiff = 0.0;
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
		numeric.poweq(grad, 2);	// grad is now gradSq
		numeric.addeq(sumGradSq, grad);
		var weightedGradSq = numeric.mul(grad, scoreDiff)
		numeric.addeq(sumWeightedGradSq, weightedGradSq);
	}
	return {
		sumGrad: sumGrad,
		sumGradSq: sumGradSq,
		sumWeightedGrad: sumWeightedGrad,
		sumWeightedGradSq: sumWeightedGradSq,
		sumScoreDiff: sumScoreDiff
	}
}

// ----------------------------------------------------------------------------


// Wait for messages
process.on('message', function(e) {
	var msg = JSON.parse(e);
	if (msg.command === 'init') {
		init(msg.source, msg.verbosity);
	} else
	if (msg.command === 'params') {
		params = msg.params;
	} else
	if (msg.command === 'sample') {
		var stats = sample(msg.nSamples);
		process.send(JSON.stringify(stats));
	}
})




