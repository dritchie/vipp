var numeric = require('numeric');
var assert = require('assert');
var _ = require('underscore');


// Just some convenience functions for tensors that numeric.js
//    doesn't already provide.

function isscalar(dim) {
	return dim.length === 0 || dim[0] === undefined;
}

function create(dim, fn) {
	if (isscalar(dim))
		return fn();
	else {
		var x = numeric.rep(dim, 0);
		mapeq(x, fn);
		return x;
	}
}

function map(tensor, fn) {
	var dim = numeric.dim(tensor);
	if (isscalar(dim))
		return fn(tensor);
	else if (dim.length === 1)
		return tensor.map(fn);
	else
		return numeric._foreach2(tensor, dim, 0, function(x) { return x.map(fn); });
}

function mapeq(tensor, fn) {
	var dim = numeric.dim(tensor);
	assert(!isscalar(dim), 'tensor.mapeq does not apply to scalar arguments');
	if (dim.length == 1) {
		for (var i = 0; i < tensor.length; i++)
			tensor[i] = fn(tensor[i]);
	} else {
		numeric._foreach(tensor, dim, 0, function(x) {
			for (var i = 0; i < x.length; i++)
				x[i] = fn(x[i]);
		});
	}
}

function foreach(tensor, fn) {
	var dim = numeric.dim(tensor);
	if (isscalar(dim))
		return fn(tensor);
	else if (dim.length === 1) {
		for (var i = 0; i < tensor.length; i++)
			fn(tensor[i]);
	} else {
		numeric._foreach(tensor, dim, 0, function(x) {
			for (var i = 0; i < x.length; i++)
				fn(x[i]);
		});
	}
}

function any(tensor, pred) {
	var test = false;
	foreach(tensor, function(x) {
		test |= pred(x);
	});
	return test;
}

function all(tensor, pred) {
	var test = true;
	foreach(tensor, function(x) {
		test &= pred(x);
	});
	return test;
}

var maxreduce = numeric.mapreduce('accum = Math.max(xi, accum)', '-Infinity');

module.exports = {
	create: create,
	map: map,
	mapeq: mapeq,
	foreach: foreach,
	any: any,
	all: all,
	maxreduce: maxreduce
};


