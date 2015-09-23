var numeric = require('numeric');
var assert = require('assert');
var _ = require('underscore');


// Just some convenience functions for tensors that numeric.js
//    doesn't already provide.

function isscalar(dim) {
	return dim.length === 0 || dim[0] === undefined;
}

function getdim(tensor) {
	var dim = numeric.dim(tensor);
	// tensors containing AD tapes will have an extra 'undefined' in the dim array
	if (dim.length > 0 && dim[dim.length-1] === undefined)
		dim.splice(dim.length-1, 1);
	return dim;
}

function create(dim, fn) {
	if (dim.length === 0)
		return fn();
	else {
		var x = numeric.rep(dim, 0);
		mapeq(x, fn);
		return x;
	}
}

function map(tensor, fn) {
	var dim = getdim(tensor);
	if (dim.length === 0)
		return fn(tensor);
	else if (dim.length === 1)
		return tensor.map(fn);
	else {
		return numeric._foreach2(tensor, dim, 0, function(x) { return x.map(fn); });
	}
}

function map2(t1, t2, fn) {
	var dim1 = getdim(t1);
	var dim2 = getdim(t2);
	assert(numeric.same(dim1, dim2));
	if (dim1.length === 0)
		return fn(t1, t2);
	else {
		return numeric._biforeach2(t1, t2, dim1, 0, function(x, y) {
			var ret = Array(x.length);
			for (var i = 0; i < x.length; i++)
				ret[i] = fn(x[i], y[i]);
			return ret;
		});
	}
}

function mapeq(tensor, fn) {
	var dim = getdim(tensor);
	assert(dim.length !== 0, 'tensor.mapeq does not apply to scalar arguments');
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
	var dim = getdim(tensor);
	if (dim.length === 0)
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
	map2: map2,
	mapeq: mapeq,
	foreach: foreach,
	any: any,
	all: all,
	maxreduce: maxreduce
};


