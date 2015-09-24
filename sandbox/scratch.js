// var mu = -1;
// var sigma = 1.5;
// var x = 0;

// var erp = require('src/erp.js');

// var trueGrad = erp.gaussianERP.grad([mu, sigma], x);
// console.log('true grad:', trueGrad);

// var adGradFn = ad_gradientR(function(params) {
// 	return erp.gaussianERP.score(params, x);
// });
// var adGrad = adGradFn([mu, sigma]);
// console.log('  ad grad:', adGrad);


// ----------------------------------------------------------------------------

// var a = 2;
// var b = 2;
// var x = 0.5;

// var erp = require('src/erp.js');

// var trueGrad = erp.gammaERP.grad([a, b], x);
// console.log('true grad:', trueGrad);

// var adGradFn = ad_gradientR(function(params) {
// 	return erp.gammaERP.score(params, x);
// });
// var adGrad = adGradFn([a, b]);
// console.log('  ad grad:', adGrad);


// ----------------------------------------------------------------------------


// var sigmoid = function(x) {
// 	return 1 / (1 + Math.exp(-x));
// }
// var logit = function(x) {
// 	return Math.log(x / (1 - x));
// }

// var target = function() {
// 	var x = flip(0.5);
// 	factor(x ? 0 : -10);
// 	return x;
// }

// var guide = function(params) {
// 	// var p = param(params, 0.5);
// 	// return flip(p);

// 	// var p = param(params, Math.log(0.5));
// 	// var theta = Math.exp(p);

// 	var p = param(params, logit(0.5));
// 	var theta = sigmoid(p);

// 	return flip(theta);
// }


// ----------------------------------------------------------------------------


// var target = function() {
// 	var x = gaussian(0, 1);
// 	factor(gaussianERP.score([0.5, 0.1], x));
// 	return x;
// }

// var guide = function(params) {
// 	var mu = param(params, 0);
// 	// var sigma = Math.exp(param(params, 0));
// 	var sigma = param(params, 1);
// 	return gaussian(mu, sigma);
// }


// ----------------------------------------------------------------------------


// var target = function() {
// 	var x = gamma(1, 1);
// 	var y = gamma(1, 1);
// 	factor(gammaERP.score([2, 2], x));
// 	factor(gammaERP.score([2, 2], y));
// 	return x + y;
// }

// var guide = function(params) {
// 	var x = gamma(param(params, 1), param(params, 1));
// 	var y = gamma(param(params, 1), param(params, 1));
// 	return x + y;
// }


// ----------------------------------------------------------------------------

// var target = function() {
// 	var x = gaussian(0, 1);
// 	var y = gaussian(0, 1);
// 	factor(gaussianERP.score([1, 0.5], x));
// 	factor(gaussianERP.score([1, 0.5], y));
// 	return x + y;
// }

// var guide = function(params) {
// 	var x = gaussian(param(params, 0), param(params, 1));
// 	var y = gaussian(param(params, 0), param(params, 1));
// 	return x + y;
// }

// ----------------------------------------------------------------------------

// var target = function() {
// 	var x = gaussian(0, 1);
// 	var y = gamma(1, 1);
// 	factor(gaussianERP.score([1, 0.5], x));
// 	factor(gammaERP.score([2, 2], y));
// 	return x + y;
// }

// var guide = function(params) {
// 	var x = gaussian(param(params, 0), param(params, 1));
// 	var y = gamma(param(params, 1), param(params, 1));
// 	return x + y;
// }

// ----------------------------------------------------------------------------


// var a0 = 1;
// var b0 = 1;
// var mu0 = 0;
// var lambda0 = 1;

// var target = function() {
// 	var tau = gamma(a0, 1.0 / b0);
// 	var mu = gaussian(mu0, 1.0 / Math.sqrt(lambda0 * tau));
// 	factor(gaussianERP.score([mu, 1.0 / Math.sqrt(tau)], 1.0));
// 	factor(gaussianERP.score([mu, 1.0 / Math.sqrt(tau)], 2.0));
// 	return mu;
// }

// // // Mean-field
// // var guide = function(params) {
// // 	var tau = gamma(param(params, a0), param(params, 1 / b0));
// // 	var mu = gaussian(param(params, mu0), param(params, 1 / Math.sqrt(lambda0*tau)));
// // 	return mu;
// // }

// // // Mean-field w/ bounds
// // var guide = function(params) {
// // 	var tau = gamma(Math.exp(param(params, Math.log(a0))), Math.exp(param(params, Math.log(1 / b0))));
// // 	var mu = gaussian(param(params, mu0), Math.exp(param(params, Math.log(1 / Math.sqrt(lambda0*tau)))));
// // 	return mu;
// // }

// // Mean-field + backprop(?)
// var guide = function(params) {
// 	var tau = gamma(param(params, a0), param(params, 1 / b0));
// 	var mu = gaussian(mu0 + param(params, 0.0), (1 / Math.sqrt(lambda0*tau)) * param(params, 1.0));
// 	return mu;
// }

// ----------------------------------------------------------------------------

var targetsum = 10.0;

var target = function() {
	var x1 = gaussian(0.0, 5.0);
	var x2 = gaussian(0.0, 5.0);
	var x3 = gaussian(0.0, 5.0);
	var sum = x1 + x2 + x3;
	factor(gaussianERP.score([targetsum, 0.1], sum));
	return [x1, x2, x3];
};

// // Mean-field
// var guide = function(params) {
// 	var p = function(x) { return param(params, x); };
// 	var x1 = gaussian(p(0.0), p(5.0));
// 	var x2 = gaussian(p(0.0), p(5.0));
// 	var x3 = gaussian(p(0.0), p(5.0));
// 	return [x1, x2, x3];
// }

// // Mean-field w/ random initialization
// var guide = function(params) {
// 	var mup = function() { return param(params, undefined, undefined, gaussianERP.sample, [0.0, 1.0]); };
// 	// var sigp = function() { return param(params, undefined, undefined, gammaERP.sample, [1.0, 1.0]); };
// 	var sigp = function() { return Math.abs(param(params, undefined, undefined, gammaERP.sample, [1.0, 1.0])); };
// 	var x1 = gaussian(mup(), sigp());
// 	var x2 = gaussian(mup(), sigp());
// 	var x3 = gaussian(mup(), sigp());
// 	return [x1, x2, x3];
// }

// // Context-sensitive
// var guide = function(params) {
// 	var p = function(x) { return param(params, x); };
// 	var x1 = gaussian(p(0.0), p(5.0));
// 	var x2 = gaussian(p(0.0) + p(0.0)*x1, p(5.0));
// 	var x3 = gaussian(p(0.0) + p(0.0)*x1 + p(0.0)*x2, p(5.0));
// 	return [x1, x2, x3];
// }

var bounds = require('src/boundsTransforms');

// Context-sensitive w/ random initialization
var guide = function(params) {
	var p1 = function() { return param(params, undefined, undefined, gaussianERP.sample, [0.0, 1.0]); };
	var p2 = function() { return param(params, undefined, bounds.nonNegative, gammaERP.sample, [1.0, 1.0]); };
	var x1 = gaussian(p1(), p2());
	var x2 = gaussian(p1() + p1()*x1, p2());
	var x3 = gaussian(p1() + p1()*x1 + p1()*x2, p2());
	return [x1, x2, x3];
};

// ----------------------------------------------------------------------------


console.time('time');
var result = variational.infer(target, guide, undefined, {
	verbosity: 4,
	// nSamples: 1,
	nSamples: 2,
	nSteps: 1,
	// nSteps: 20,
	convergeEps: 0.1,
	// convergeEps: 0.01,
	// initLearnRate: 0.5
	// initLearnRate: 0.25,
	// initLearnRate: 0.1
	// recordStepStats: true
});
console.log(result.params);
console.log('Samples from guide:');
for (var i = 0; i < 20; i++) {
	var nums = guide(result.params);
	var sum = nums[0] + nums[1] + nums[2];
	console.log(sum + ': ' + nums);
}
// return result;







