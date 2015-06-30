var targetsum = 10.0

var target = function() {
	var x1 = gaussian(0.0, 5.0);
	var x2 = gaussian(0.0, 5.0);
	var x3 = gaussian(0.0, 5.0);
	var sum = x1 + x2 + x3;
	factor(gaussianERP.score([targetsum, 0.1], sum));
	return sum;
}

// Mean-field
var guide_meanField = function(params) {
	var p = function(x) { return param(params, x); };
	var x1 = gaussian(p(0.0), p(5.0));
	var x2 = gaussian(p(0.0), p(5.0));
	var x3 = gaussian(p(0.0), p(5.0));
	return x1 + x2 + x3;
}

// Context-sensitive
var guide_context = function(params) {
	var p = function(x) { return param(params, x); };
	var x1 = gaussian(p(0.0), p(5.0));
	var x2 = gaussian(p(0.0) + p(0.0)*x1, p(5.0));
	var x3 = gaussian(p(0.0) + p(0.0)*x1 + p(0.0)*x2, p(5.0));
	return x1 + x2 + x3;
}