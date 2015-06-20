import macros from './ad/macros'
__initAD;

'use strict';

// TODO: In this and all other autodiff-ed files:
// Can the +=, -=, etc. operators be overloaded?
//    If not, we must take care never to use these in
//    autodiff-ed code...

var coroutine = null;

// target: original probabilistic program
// guide: variational program
// args: inputs to program (i.e. observed evidence)
// opts: options controlling inference behavior
// Returns the inferred variational params
function variational(target, guide, args, opts) {
	//
}

function sample(erp, params) {
	//
}

function factor(num) {
	//
}

// Create/lookup a param
function param(params, initialVal) {
	//
}

// Create/lookup a param that has a prior
function paramWithPrior(params, initialVal, scoreFn, hypers) {
	//
}

module.exports = {
	run: variational,
	sample: sample,
	factor: factor,
	param: param,
	paramWithPrior: paramWithPrior
};