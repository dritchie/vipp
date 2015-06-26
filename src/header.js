'use strict';

var erp = require('src/erp.js');
// // TEST: no AD
// var erp = require('src/erp.sjs');
// //
var variational = require('src/variational.js');

module.exports = {};

for (var prop in erp)
	module.exports[prop] = erp[prop];
for (var prop in variational)
	module.exports[prop] = variational[prop];	


// Common ERPs
var flip = function(theta) {
  return sample(erp.bernoulliERP, [theta]);
};

var randomInteger = function(n) {
  return sample(erp.randomIntegerERP, [n]);
};

var discrete = function(n) {
  return sample(erp.discreteERP, [n]);
};

var categorical = function(ps, vs) {
  return vs[discrete(ps)];
}

var gaussian = function(mu, sigma) {
  return sample(erp.gaussianERP, [mu, sigma]);
};

// var multivariateGaussian = function(mu, cov) {
//   return sample(multivariateGaussianERP, [mu, cov]);
// };

var uniform = function(a, b) {
  return sample(erp.uniformERP, [a, b]);
};

var uniformDraw = function(l) {
  return l[sample(erp.randomIntegerERP, [l.length])];
};

var dirichlet = function(alpha) {
  return sample(erp.dirichletERP, alpha);
};

var poisson = function(mu, k) {
  return sample(erp.poissonERP, [mu, k]);
};

var binomial = function(p, n) {
  return sample(erp.binomialERP, [p, n]);
};

var beta = function(a, b) {
  return sample(erp.betaERP, [a, b]);
};

var exponential = function(a) {
  return sample(erp.exponentialERP, [a]);
};

var gamma = function(shape, scale) {
  return sample(erp.gammaERP, [shape, scale]);
};

// var deltaERP = function(v) {
//   return erp.makeDeltaERP(v);
// }

// var multiplexERP = function(vs, erps) {
//   return erp.makeMultiplexERP(vs, erps);
// }

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
