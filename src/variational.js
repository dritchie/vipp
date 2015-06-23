'use strict';

// TODO: In this and all other autodiff-ed files:
// Can the +=, -=, etc. operators be overloaded?
//    If not, we must take care never to use these in
//    autodiff-ed code...

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
	function opt(val, defaultval) {
		return val === undefined ? defaultval : val;
	}
	var nSteps = opt(opts.nSteps, 100);
	var nSamples = opt(opts.nSamples, 100);
	var initLearnRate = opt(opts.initLearnRate, 1);
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
		rerun: function() {
			this.choiceIndex = 0;
			this.score = 0;	
			return thunk();
		}
	}

	// Install coroutine
	var oldCoroutine = coroutine;
	coroutine = vo;

	var params = [];

	// Thunks that we'll feed to 'run' and 'rerun'
	var targetThunk = function() {
		return target(args);
	}
	var guideGrad = ad_gradientR(function(p) {
		guide(args, p);
		return vco.score;
	})
	var guideGradThunk = function() {
		return guideGrad(params);
	}

	// Do variational inference
	var currStep = 0;
	do {
		for (var s = 0; s < nSamples; s++) {
			var grad = vco.run(guideGradThunk);
			var guideScore = vco.score;
			vo.rerun(targetThunk);
			var targetScore = vco.score;
			// TODO: Finish this.
		}
		currStep++;
	} while (convergeMeasure > convergeEps && currStep < nSteps);

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



