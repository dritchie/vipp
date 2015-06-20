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

import macros from './ad/macros'
__initAD;

'use strict';

var numeric = require('numeric');
var _ = require('underscore');
var assert = require('assert');
var util = require('./util.sjs');

var LOG_2PI = 1.8378770664093453;

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
    function uniformScore(params, val) {
      if (val < params[0] || val > params[1]) {
        return -Infinity;
      }
      return -Math.log(params[1] - params[0]);
    }
    );

var bernoulliERP = new ERP(
    function flipSample(params) {
      var weight = params[0];
      var val = Math.random() < weight;
      return val;
    },
    function flipScore(params, val) {
      if (val != true && val != false) {
        return -Infinity;
      }
      var weight = params[0];
      return val ? Math.log(weight) : Math.log(1 - weight);
    },
    {
      support: function flipSupport(params) {
        return [true, false];
      },
      grad: function flipGrad(params, val) {
        var weight = params[0];
        assert(weight >= 0 && weight <= 1,
               'bernoulliERP param outside of domain.');
        return val ? [1 / weight] : [-1 / (1 - weight)];
      }
    }
    );


var randomIntegerERP = new ERP(
    function randomIntegerSample(params) {
      return Math.floor(Math.random() * params[0]);
    },
    function randomIntegerScore(params, val) {
      var stop = params[0];
      var inSupport = (val == Math.floor(val)) && (0 <= val) && (val < stop);
      return inSupport ? -Math.log(stop) : -Infinity;
    },
    {
      support: function randomIntegerSupport(params) {
        return _.range(params[0]);
      }
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

function gaussianScore(params, x) {
  var mu = params[0];
  var sigma = params[1];
  return -0.5 * (LOG_2PI + 2 * Math.log(sigma) + (x - mu) * (x - mu) / (sigma * sigma));
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

var gaussianERP = new ERP(gaussianSample, gaussianScore, { grad: gaussianGrad });

function multivariateGaussianSample(params) {
  var mu = params[0];
  var cov = params[1];
  var xs = mu.map(function() {return gaussianSample([0, 1])});
  var svd = numeric.svd(cov);
  var scaledV = numeric.transpose(svd.V).map(function(x) {return numeric.mul(numeric.sqrt(svd.S), x)});
  xs = numeric.dot(xs, numeric.transpose(scaledV));
  return numeric.add(xs, mu);
}

function multivariateGaussianScore(params, x) {
  var mu = params[0];
  var cov = params[1];
  var n = mu.length;
  var coeffs = n * LOG_2PI + Math.log(numeric.det(cov));
  var xSubMu = numeric.sub(x, mu);
  var exponents = numeric.dot(numeric.dot(xSubMu, numeric.inv(cov)), xSubMu);
  return -0.5 * (coeffs + exponents);
}

var multivariateGaussianERP = new ERP(multivariateGaussianSample, multivariateGaussianScore);

var discreteERP = new ERP(
    function discreteSample(params) {
      return multinomialSample(params[0]);
    },
    function discreteScore(params, val) {
      var probs = util.normalizeArray(params[0]);
      var stop = probs.length;
      var inSupport = (val == Math.floor(val)) && (0 <= val) && (val < stop);
      return inSupport ? Math.log(probs[val]) : -Infinity;
    },
    {
      support:
          function discreteSupport(params) {
            return _.range(params[0].length);
          }
    }
    );

var gammaCof = [
  76.18009172947146,
  -86.50532032941677,
  24.01409824083091,
  -1.231739572450155,
  0.1208650973866179e-2,
  -0.5395239384953e-5];

function logGamma(xx) {
  var x = xx - 1.0;
  var tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  var ser = 1.000000000190015;
  for (var j = 0; j <= 5; j++) {
    x++;
    ser += gammaCof[j] / x;
  }
  return -tmp + Math.log(2.5066282746310005 * ser);
}

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
      psi -= 1 / (x + i);
    }
    return psi;
  } else {
    var psi = Math.log(x) - 1 / (2 * x);
    var invsq = 1 / (x * x);
    var z = 1;
    for (var i = 0; i < digammaCof.length; i++) {
      z *= invsq;
      psi += digammaCof[i] * z;
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
    function gammaScore(params, val) {
      var a = params[0];
      var b = params[1];
      var x = val;
      return (a - 1) * Math.log(x) - x / b - logGamma(a) - a * Math.log(b);
    },
    {
      grad: gammaGrad
    }
    );

var exponentialERP = new ERP(
    function exponentialSample(params) {
      var a = params[0];
      var u = Math.random();
      return Math.log(u) / (-1 * a);
    },
    function exponentialScore(params, val) {
      var a = params[0];
      return Math.log(a) - a * val;
    }
    );

function logBeta(a, b) {
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

function betaSample(params) {
  var a = params[0];
  var b = params[1];
  var x = gammaSample([a, 1]);
  return x / (x + gammaSample([b, 1]));
}

var betaERP = new ERP(
    betaSample,
    function betaScore(params, val) {
      var a = params[0];
      var b = params[1];
      var x = val;
      return ((x > 0 && x < 1) ?
          (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logBeta(a, b) :
          -Infinity);
    },
    {
      grad: function(params, x) {
        var a = params[0];
        var b = params[1];
        assert(a > 0 && b > 0, 'betaERP param outside of domain.');
        var d = digamma(a + b);
        return [Math.log(x) - digamma(a) + d, Math.log(1 - x) - digamma(b) + d];
      }
    }
    );

function binomialG(x) {
  if (x === 0) {
    return 1;
  }
  if (x === 1) {
    return 0;
  }
  var d = 1 - x;
  return (1 - (x * x) + (2 * x * Math.log(x))) / (d * d);
}

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
      p /= x;
    }
    else {
      k += a;
      n = b - 1;
      p = (p - x) / (1 - x);
    }
  }
  var u;
  for (var i = 0; i < n; i++) {
    u = Math.random();
    if (u < p) {
      k++;
    }
  }
  return k || 0;
}

var binomialERP = new ERP(
    binomialSample,
    function binomialScore(params, val) {
      var p = params[0];
      var n = params[1];
      if (n > 20 && n * p > 5 && n * (1 - p) > 5) {
        // large n, reasonable p approximation
        var s = val;
        var inv2 = 1 / 2;
        var inv3 = 1 / 3;
        var inv6 = 1 / 6;
        if (s >= n) {
          return -Infinity;
        }
        var q = 1 - p;
        var S = s + inv2;
        var T = n - s - inv2;
        var d1 = s + inv6 - (n + inv3) * p;
        var d2 = q / (s + inv2) - p / (T + inv2) + (q - inv2) / (n + 1);
        d2 = d1 + 0.02 * d2;
        var num = 1 + q * binomialG(S / (n * p)) + p * binomialG(T / (n * q));
        var den = (n + inv6) * p * q;
        var z = num / den;
        var invsd = Math.sqrt(z);
        z = d2 * invsd;
        return gaussianScore([0, 1], z) + Math.log(invsd);
      } else {
        // exact formula
        return (lnfact(n) - lnfact(n - val) - lnfact(val) +
            val * Math.log(p) + (n - val) * Math.log(1 - p));
      }
    },
    {
      support:
          function binomialSupport(params) {
            return _.range(params[1]).concat([params[1]]);
          }
    }
    );

function fact(x) {
  var t = 1;
  while (x > 1) {
    t *= x--;
  }
  return t;
}

function lnfact(x) {
  if (x < 1) {
    x = 1;
  }
  if (x < 12) {
    return Math.log(fact(Math.round(x)));
  }
  var invx = 1 / x;
  var invx2 = invx * invx;
  var invx3 = invx2 * invx;
  var invx5 = invx3 * invx2;
  var invx7 = invx5 * invx2;
  var sum = ((x + 0.5) * Math.log(x)) - x;
  sum += Math.log(2 * Math.PI) / 2;
  sum += (invx / 12) - (invx3 / 360);
  sum += (invx5 / 1260) - (invx7 / 1680);
  return sum;
}

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
          mu -= x;
          k += m;
        }
      }
      var emu = Math.exp(-mu);
      var p = 1;
      do {
        p *= Math.random();
        k++;
      } while (p > emu);
      return (k - 1) || 0;
    },
    function poissonScore(params, val) {
      var mu = params[0];
      var k = val;
      return k * Math.log(mu) - mu - lnfact(k);
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
    theta[j] /= ssum;
  }
  return theta;
}

function dirichletScore(params, val) {
  var alpha = params;
  var theta = val;
  var asum = 0;
  for (var i = 0; i < alpha.length; i++) {
    asum += alpha[i];
  }
  var logp = logGamma(asum);
  for (var j = 0; j < alpha.length; j++) {
    logp += (alpha[j] - 1) * Math.log(theta[j]);
    logp -= logGamma(alpha[j]);
  }
  return logp;
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

var dirichletERP = new ERP(dirichletSample, dirichletScore, { grad: dirichletGrad });

function multinomialSample(theta) {
  var thetaSum = util.sum(theta);
  var x = Math.random() * thetaSum;
  var k = theta.length;
  var probAccum = 0;
  for (var i = 0; i < k; i++) {
    probAccum += theta[i];
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
  multivariateGaussianERP: multivariateGaussianERP,
  poissonERP: poissonERP,
  randomIntegerERP: randomIntegerERP,
  uniformERP: uniformERP,
  makeMarginalERP: makeMarginalERP,
  makeDeltaERP: makeDeltaERP,
  makeMultiplexERP: makeMultiplexERP
};
