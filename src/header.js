'use strict';

var erp = require('src/erp.js');
var variational = require('src/variational.js');

module.exports = {};

for (var prop in erp)
	module.exports[prop] = erp[prop];
for (var prop in variational)
	module.exports[prop] = variational[prop];	


// Common ERPs
var flip = function(theta) {
  return sample(bernoulliERP, [theta]);
};

var randomInteger = function(n) {
  return sample(randomIntegerERP, [n]);
};

var discrete = function(n) {
  return sample(discreteERP, [n]);
};

var categorical = function(ps, vs) {
  return vs[discrete(ps)];
}

var gaussian = function(mu, sigma) {
  return sample(gaussianERP, [mu, sigma]);
};

// var multivariateGaussian = function(mu, cov) {
//   return sample(multivariateGaussianERP, [mu, cov]);
// };

var uniform = function(a, b) {
  return sample(uniformERP, [a, b]);
};

var uniformDraw = function(l) {
  return l[sample(randomIntegerERP, [l.length])];
};

var dirichlet = function(alpha) {
  return sample(dirichletERP, alpha);
};

var poisson = function(mu, k) {
  return sample(poissonERP, [mu, k]);
};

var binomial = function(p, n) {
  return sample(binomialERP, [p, n]);
};

var beta = function(a, b) {
  return sample(betaERP, [a, b]);
};

var exponential = function(a) {
  return sample(exponentialERP, [a]);
};

var gamma = function(shape, scale) {
  return sample(gammaERP, [shape, scale]);
};

// var deltaERP = function(v) {
//   return top.makeDeltaERP(v);
// }

// var multiplexERP = function(vs, erps) {
//   return top.makeMultiplexERP(vs, erps);
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
