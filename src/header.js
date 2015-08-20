'use strict';

var erp = require('./erp.js');
var variational = require('./variational.js');

module.exports = {};

for (var prop in erp)
	module.exports[prop] = erp[prop];
for (var prop in variational)
	module.exports[prop] = variational[prop];	


// Common ERPs
var flip = function(name, theta) {
  return sample(name, erp.bernoulliERP, [theta]);
};

var randomInteger = function(name, n) {
  return sample(name, erp.randomIntegerERP, [n]);
};

var discrete = function(name, theta) {
  return sample(name, erp.discreteERP, theta);
};

var categorical = function(name, ps, vs) {
  return vs[discrete(name, ps)];
}

var gaussian = function(name, mu, sigma) {
  return sample(name, erp.gaussianERP, [mu, sigma]);
};

var uniform = function(name, a, b) {
  return sample(name, erp.uniformERP, [a, b]);
};

var uniformDraw = function(name, l) {
  return l[sample(name, erp.randomIntegerERP, [l.length])];
};

var dirichlet = function(name, alpha) {
  return sample(name, erp.dirichletERP, alpha);
};

var poisson = function(name, mu, k) {
  return sample(name, erp.poissonERP, [mu, k]);
};

var binomial = function(name, p, n) {
  return sample(name, erp.binomialERP, [p, n]);
};

var beta = function(name, a, b) {
  return sample(name, erp.betaERP, [a, b]);
};

var exponential = function(name, a) {
  return sample(name, erp.exponentialERP, [a]);
};

var gamma = function(name, shape, scale) {
  return sample(name, erp.gammaERP, [shape, scale]);
};


module.exports.flip = flip;
module.exports.randomInteger = randomInteger;
module.exports.discrete = discrete;
module.exports.categorical = categorical;
module.exports.gaussian = gaussian;
module.exports.uniform = uniform;
module.exports.uniformDraw = uniformDraw;
module.exports.dirichlet = dirichlet;
module.exports.poisson = poisson;
module.exports.binomial = binomial;
module.exports.beta = beta;
module.exports.exponential = exponential;
module.exports.gamma = gamma;
