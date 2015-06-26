// Example from https://en.wikipedia.org/wiki/Variational_Bayesian_methods

var a0 = 1;
var b0 = 1;
var mu0 = 0;
var lambda0 = 1;

var target = function() {
	var tau = gamma(a0, 1 / b0);
	var mu = gaussian(mu0, 1 / Math.sqrt(lambda0 * tau));
	factor(gaussianERP.score([mu, 1 / Math.sqrt(tau)], 1));
	factor(gaussianERP.score([mu, 1 / Math.sqrt(tau)], 2));
	return mu;
}

var guide = function(params) {
	var tau = gamma(param(params, a0), param(params, 1 / b0));
	var mu = gaussian(param(params, mu0), param(params, 1 / Math.sqrt(lambda0*tau)));
	return mu;
}


return infer(target, guide, undefined, {
	verbosity: 2,
	nSamples: 100,
	nSteps: 5000,
	convergeEps: 0.1,
	initLearnRate: 0.5
});