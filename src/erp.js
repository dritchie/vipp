////////////////////////////////////////////////////////////////////
// ERPs
//
// Elementary Random Primitives (ERPs) are the representation of
// distributions. They can have sampling, scoring, and support
// functions. A single ERP need not hve all three, but some inference
// functions will complain if they're missing one.
//
// The main thing we can do with ERPs in WebPPL is feed them into the
// "sample" primitive to get a sample. At top level we will also have
// some "inspection" functions to visualize them?
//
// required:
// - erp.sample(params) returns a value sampled from the distribution.
// - erp.score(params, val) returns the log-probability of val under the distribution.
//
// optional:
// - erp.support(params) gives an array of support elements.
// - erp.grad(params, val) gives the gradient of score at val wrt params.
// - erp.proposer is an erp for making mh proposals conditioned on the previous value

'use strict';

// var numeric = require('numeric');
var _ = require('underscore');
var assert = require('assert');
var util = require('./util.js');
var scorers = require('./erp_scorers.adjs');
var adscorers = require('./erp_scorers.js');

function ERP(sampler, scorer, auxParams) {
  auxParams = typeof auxParams === 'undefined' ? {} : auxParams;
  this.sample = sampler;
  this.score = scorer;
  for (var key in auxParams) {
    if (auxParams.hasOwnProperty(key)) {
      this[key] = auxParams[key];
    }
  }
}

var uniformERP = new ERP(
    function uniformSample(params) {
      var u = Math.random();
      return (1 - u) * params[0] + u * params[1];
    },
    scorers.uniform,
    {
      adscore: adscorers.uniform
    }
    );

var bernoulliERP = new ERP(
    function flipSample(params) {
      var weight = params[0];
      var val = Math.random() < weight;
      return val;
    },
    scorers.flip,
    {
      support: function flipSupport(params) {
        return [true, false];
      },
      grad: function flipGrad(params, val) {
        var weight = params[0];
        assert(weight >= 0 && weight <= 1,
               'bernoulliERP param outside of domain.');
        return val ? [1 / weight] : [-1 / (1 - weight)];
      },
      adscore: adscorers.flip,
      adentropy: adscorers.flipEntropy
    }
    );


var randomIntegerERP = new ERP(
    function randomIntegerSample(params) {
      return Math.floor(Math.random() * params[0]);
    },
    scorers.randomInteger,
    {
      support: function randomIntegerSupport(params) {
        return _.range(params[0]);
      },
      adscore: adscorers.randomInteger
    }
    );

function gaussianSample(params) {
  var mu = params[0];
  var sigma = params[1];
  var u, v, x, y, q;
  do {
    u = 1 - Math.random();
    v = 1.7156 * (Math.random() - 0.5);
    x = u - 0.449871;
    y = Math.abs(v) + 0.386595;
    q = x * x + y * (0.196 * y - 0.25472 * x);
  } while (q >= 0.27597 && (q > 0.27846 || v * v > -4 * u * u * Math.log(u)));
  return mu + sigma * v / u;
}

function gaussianGrad(params, x) {
  var mu = params[0];
  var sigma = params[1];
  assert(sigma > 0, 'gaussianERP param outside of domain');
  var sigma2 = sigma * sigma;
  var xdiff = x - mu;
  var muGrad = xdiff / sigma2;
  var sigmaGrad = (xdiff * xdiff / sigma2 - 1) / sigma;
  return [muGrad, sigmaGrad];
}

var gaussianERP = new ERP(gaussianSample, scorers.gaussian, {
  grad: gaussianGrad,
  adscore: adscorers.gaussian
});

var discreteERP = new ERP(
    function discreteSample(params) {
      return multinomialSample(params);
    },
    scorers.discrete,
    {
      support:
          function discreteSupport(params) {
            return _.range(params.length);
          },
      adscore: adscorers.discrete
    }
    );

var digammaCof = [
  -1 / 12, 1 / 120, -1 / 252, 1 / 240,
  -5 / 660, 691 / 32760, -1 / 12];

function digamma(x) {
  if (x < 0) {
    return digamma(1 - x) - Math.PI / Math.tan(Math.PI * x);
  } else if (x < 6) {
    var n = Math.ceil(6 - x);
    var psi = digamma(x + n);
    for (var i = 0; i < n; i++) {
      psi = psi - (1 / (x + i));
    }
    return psi;
  } else {
    var psi = Math.log(x) - 1 / (2 * x);
    var invsq = 1 / (x * x);
    var z = 1;
    for (var i = 0; i < digammaCof.length; i++) {
      z = z * invsq;
      psi = psi + (digammaCof[i] * z);
    }
    return psi;
  }
}

function gammaSample(params) {
  var a = params[0];
  var b = params[1];
  if (a < 1) {
    return gammaSample([1 + a, b]) * Math.pow(Math.random(), 1 / a);
  }
  var x, v, u;
  var d = a - 1 / 3;
  var c = 1 / Math.sqrt(9 * d);
  while (true) {
    do {
      x = gaussianSample([0, 1]);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    u = Math.random();
    if ((u < 1 - 0.331 * x * x * x * x) || (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v)))) {
      return b * d * v;
    }
  }
}

var gammaGrad = function(params, x)  {
  var a = params[0];
  var b = params[1];
  assert(a > 0 && b > 0, 'gammaERP param outside of domain.');
  var aGrad = Math.log(x / b) - digamma(a);
  var bGrad = x / (b * b) - a / b;
  return [aGrad, bGrad];
};

// params are shape and scale
var gammaERP = new ERP(
    gammaSample,
    scorers.gamma,
    {
      grad: gammaGrad,
      adscore: adscorers.gamma
    }
    );

var exponentialERP = new ERP(
    function exponentialSample(params) {
      var a = params[0];
      var u = Math.random();
      return Math.log(u) / (-1 * a);
    },
    scorers.exponential,
    {
      adscore: adscorers.exponential
    }
    );

function __betaSample(params) {
  var a = params[0];
  var b = params[1];
  var x = gammaSample([a, 1]);
  return x / (x + gammaSample([b, 1]));
}
// I can never reproduce this in simple test code, but in context, this
//    occasionally samples a value == 1. 
// Kludge: just rejection sample until we get something in the support.
function betaSample(params) {
  var x;
  do {
    x = __betaSample(params);
  } while (x <= 0 || x >= 1)
  return x;
}

var betaERP = new ERP(
    betaSample,
    scorers.beta,
    {
      grad: function(params, x) {
        var a = params[0];
        var b = params[1];
        assert(a > 0 && b > 0, 'betaERP param outside of domain.');
        var d = digamma(a + b);
        return [Math.log(x) - digamma(a) + d, Math.log(1 - x) - digamma(b) + d];
      },
      adscore: adscorers.beta,
      adentropy: adscorers.betaEntropy
    }
    );

function binomialSample(params) {
  var p = params[0];
  var n = params[1];
  var k = 0;
  var N = 10;
  var a, b;
  while (n > N) {
    a = 1 + n / 2;
    b = 1 + n - a;
    var x = betaSample([a, b]);
    if (x >= p) {
      n = a - 1;
      p = p / x;
    }
    else {
      k = k + a;
      n = b - 1;
      p = (p - x) / (1 - x);
    }
  }
  var u;
  for (var i = 0; i < n; i++) {
    u = Math.random();
    if (u < p) {
      k = k + 1;
    }
  }
  return k || 0;
}

var binomialERP = new ERP(
    binomialSample,
    scorers.binomial,
    {
      support:
          function binomialSupport(params) {
            return _.range(params[1]).concat([params[1]]);
          },
      adscore: adscorers.binomial
    }
    );

var poissonERP = new ERP(
    function poissonSample(params) {
      var mu = params[0];
      var k = 0;
      while (mu > 10) {
        var m = 7 / 8 * mu;
        var x = gammaSample([m, 1]);
        if (x > mu) {
          return (k + binomialSample([mu / x, m - 1])) || 0;
        } else {
          mu = mu - x;
          k = k + m;
        }
      }
      var emu = Math.exp(-mu);
      var p = 1;
      do {
        p = p * Math.random();
        k = k + 1;
      } while (p > emu);
      return (k - 1) || 0;
    },
    scorers.poisson,
    {
      adscore: adscorers.poisson
    }
    );

function dirichletSample(params) {
  var alpha = params;
  var ssum = 0;
  var theta = [];
  var t;
  for (var i = 0; i < alpha.length; i++) {
    t = gammaSample([alpha[i], 1]);
    theta[i] = t;
    ssum = ssum + t;
  }
  for (var j = 0; j < theta.length; j++) {
    theta[j] = theta[j] / ssum;
  }
  return theta;
}

function dirichletGrad(params, val) {
  var alpha = params;
  var d = digamma(util.sum(alpha));
  var grad = [];
  for (var i = 0; i < alpha.length; i++) {
    assert(alpha[i] > 0, 'dirichletERP param outside of domain');
    grad.push(Math.log(val[i]) - digamma(alpha[i]) + d);
  }
  return grad;
}

var dirichletERP = new ERP(dirichletSample, scorers.dirichlet, {
  grad: dirichletGrad,
  adscore: adscorers.dirichlet
});

function multinomialSample(theta) {
  var thetaSum = util.sum(theta);
  var x = Math.random() * thetaSum;
  var k = theta.length;
  var probAccum = 0;
  for (var i = 0; i < k; i++) {
    probAccum = probAccum + theta[i];
    if (probAccum >= x) {
      return i;
    } //FIXME: if x=0 returns i=0, but this isn't right if theta[0]==0...
  }
  return k;
}

// Make a discrete ERP from a {val: prob, etc.} object (unormalized).
function makeMarginalERP(marginal) {

  // Normalize distribution:
  var norm = 0;
  var supp = [];
  for (var v in marginal) {if (marginal.hasOwnProperty(v)) {
    var d = marginal[v]
    norm += d.prob;
    supp.push(d.val);
  }}
  var mapEst = {val: undefined, prob: 0};
  for (v in marginal) {if (marginal.hasOwnProperty(v)) {
    var dd = marginal[v]
    var nprob = dd.prob / norm;
    if (nprob > mapEst.prob) mapEst = {val: dd.val, prob: nprob};
    marginal[v].prob = nprob;
  }}

  // Make an ERP from marginal:
  var dist = new ERP(
      function(params) {
        var x = Math.random();
        var probAccum = 0;
        for (var i in marginal) {if (marginal.hasOwnProperty(i)) {
          probAccum += marginal[i].prob;
          // FIXME: if x=0 returns i=0, but this isn't right if theta[0]==0...
          if (probAccum >= x) return marginal[i].val;
        }}
        return marginal[i].val;
      },
      function(params, val) {
        var lk = marginal[JSON.stringify(val)];
        return lk ? Math.log(lk.prob) : -Infinity;
      },
      {
        support:
            function(params) {
              return supp;
            }
      }
      );

  dist.MAP = mapEst;
  return dist;
}

// Make an ERP that assigns probability 1 to a single value, probability 0 to everything else
var makeDeltaERP = function(v) {
  var stringifiedValue = JSON.stringify(v);
  return new ERP(
      function deltaSample(params) {
        return v;
      },
      function deltaScore(params, val) {
        if (JSON.stringify(val) === stringifiedValue) {
          return 0;
        } else {
          return -Infinity;
        }
      },
      {
        support:
            function deltaSupport(params) {
              return [v];
            }
      }
  );
};

// Make a parameterized ERP that selects among multiple (unparameterized) ERPs
var makeMultiplexERP = function(vs, erps) {
  var stringifiedVals = vs.map(JSON.stringify);
  var selectERP = function(params) {
    var stringifiedV = JSON.stringify(params[0]);
    var i = _.indexOf(stringifiedVals, stringifiedV);
    if (i === -1) {
      return undefined;
    } else {
      return erps[i];
    }
  };
  return new ERP(
      function multiplexSample(params) {
        var erp = selectERP(params);
        assert.notEqual(erp, undefined);
        return erp.sample();
      },
      function multiplexScore(params, val) {
        var erp = selectERP(params);
        if (erp === undefined) {
          return -Infinity;
        } else {
          return erp.score([], val);
        }
      },
      {
        support: function multiplexSupport(params) {
          var erp = selectERP(params);
          return erp.support();
        }
      }
  );
};

module.exports = {
  ERP: ERP,
  bernoulliERP: bernoulliERP,
  betaERP: betaERP,
  binomialERP: binomialERP,
  dirichletERP: dirichletERP,
  discreteERP: discreteERP,
  exponentialERP: exponentialERP,
  gammaERP: gammaERP,
  gaussianERP: gaussianERP,
  multinomialSample: multinomialSample,
  poissonERP: poissonERP,
  randomIntegerERP: randomIntegerERP,
  uniformERP: uniformERP,
  makeMarginalERP: makeMarginalERP,
  makeDeltaERP: makeDeltaERP,
  makeMultiplexERP: makeMultiplexERP
};
