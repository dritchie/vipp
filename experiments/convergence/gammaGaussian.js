var a0 = 1;
var b0 = 1;
var mu0 = 0;
var lambda0 = 1;

var target = function() {
	var tau = gamma(a0, 1.0 / b0);
	var mu = gaussian(mu0, 1.0 / Math.sqrt(lambda0 * tau));
	factor(gaussianERP.score([mu, 1.0 / Math.sqrt(tau)], 1.0));
	factor(gaussianERP.score([mu, 1.0 / Math.sqrt(tau)], 2.0));
	return mu;
}

// Mean-field
var guide_meanField = function(params) {
	var tau = gamma(param(params, a0), param(params, 1 / b0));
	var mu = gaussian(param(params, mu0), param(params, 1 / Math.sqrt(lambda0*tau)));
	return mu;
}

// Mean-field w/ bounds
var guide_bounds = function(params) {
	var tau = gamma(Math.exp(param(params, Math.log(a0))), Math.exp(param(params, Math.log(1 / b0))));
	var mu = gaussian(param(params, mu0), Math.exp(param(params, Math.log(1 / Math.sqrt(lambda0*tau)))));
	return mu;
}

// Mean-field + backprop(?)
var guide_backprop = function(params) {
	var tau = gamma(param(params, a0), param(params, 1 / b0));
	var mu = gaussian(mu0 + param(params, 0.0), (1 / Math.sqrt(lambda0*tau)) * param(params, 1.0));
	return mu;
}