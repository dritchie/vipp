"use strict";

var S_dualNumber = function(epsilon, primal, perturbation) {
  this.epsilon = epsilon;
  this.primal = primal;
  this.perturbation = perturbation;
};
var isDualNumber = function(dn) { return dn instanceof S_dualNumber };

var S_tape = function(epsilon, primal, factors, tapes) {
  this.epsilon = epsilon;
  this.primal = primal;
  this.factors = factors;
  this.tapes = tapes;
  this.fanout = 0;
  this.sensitivity = 0.0;
};
var isTape = function(t) { return t instanceof S_tape; };

// needswork: check if all operators used here are primitive or lifted
var lift_realreal_to_real = function(f, df_dx1, df_dx2, x1, x2) {
  var fn = function(x_1, x_2) {
    if (isTape(x_1)) {
      if (isTape(x_2))
        if (x_1.epsilon < x_2.epsilon)
          return new S_tape(x_2.epsilon, fn(x_1, x_2.primal), [df_dx2(x_1, x_2.primal)], [x_2])
        else if (x_2.epsilon < x_1.epsilon)
          return new S_tape(x_1.epsilon, fn(x_1.primal, x_2), [df_dx1(x_1.primal, x_2)], [x_1])
        else
          return new S_tape(x_1.epsilon,
                      fn(x_1.primal, x_2.primal),
                      [df_dx1(x_1.primal, x_2.primal), df_dx2(x_1.primal, x_2.primal)],
                      [x_1, x_2])
      else if (isDualNumber(x_2))
        if (x_1.epsilon < x_2.epsilon)
          return new S_dualNumber(x_2.epsilon,
                                fn(x_1, x_2.primal),
                                d_mul(df_dx2(x_1, x_2.primal), x_2.perturbation))
        else
          return new S_tape(x_1.epsilon, fn(x_1.primal, x_2), [df_dx1(x_1.primal, x_2)], [x_1])
      else
        return new S_tape(x_1.epsilon, fn(x_1.primal, x_2), [df_dx1(x_1.primal, x_2)], [x_1])
    }
    else if (isDualNumber(x_1)) {
      if (isDualNumber(x_2))
        if (x_1.epsilon < x_2.epsilon)
          return new S_dualNumber(x_2.epsilon,
                                fn(x_1, x_2.primal),
                                d_mul(df_dx2(x_1, x_2.primal), x_2.perturbation))
        else if (x_2.epsilon < x_1.epsilon)
          return new S_dualNumber(x_1.epsilon,
                                fn(x_1.primal, x_2),
                                d_mul(df_dx1(x_1.primal, x_2), x_1.perturbation))
        else
          return new S_dualNumber(x_1.epsilon,
                                fn(x_1.primal, x_2.primal),
                                d_add(d_mul(df_dx1(x_1.primal, x_2.primal), x_1.perturbation),
                                      d_mul(df_dx2(x_1.primal, x_2.primal), x_2.perturbation)))
      else if (isTape(x_2))
        if (x_1.epsilon < x_2.epsilon)
          return new S_tape(x_2.epsilon, fn(x_1, x_2.primal), [df_dx2(x_1, x_2.primal)], [x_2])
        else
          return new S_dualNumber(x_1.epsilon,
                                fn(x_1.primal, x_2),
                                d_mul(df_dx1(x_1.primal, x_2), x_1.perturbation))
      else
        return new S_dualNumber(x_1.epsilon,
                              fn(x_1.primal, x_2),
                              d_mul(df_dx1(x_1.primal, x_2), x_1.perturbation))
    }
    else {
      if (isTape(x_2))
        return new S_tape(x_2.epsilon, fn(x_1, x_2.primal), [df_dx2(x_1, x_2.primal)], [x_2])
      else if (isDualNumber(x_2))
        return new S_dualNumber(x_2.epsilon,
                              fn(x_1, x_2.primal),
                              d_mul(df_dx2(x_1, x_2.primal), x_2.perturbation))
      else
        return f(x_1, x_2)
    }
  };
  return fn(x1, x2);
};

var lift_real_to_real = function(f, df_dx, x) {
  var fn = function(x1) {
    if (isTape(x1))
      return new S_tape(x1.epsilon, fn(x1.primal), [df_dx(x1.primal)], [x1]);
    else if (isDualNumber(x1))
      return new S_dualNumber(x1.epsilon, fn(x1.primal), d_mul(df_dx(x1.primal), x1.perturbation));
    else
      return f(x1);
  }
  return fn(x);
};

/** Lifting operations **/
function isNumeric(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

/** functional wrappers for primitive operators **/
var f_minus = function(a) {return -a};

var f_add = function(a,b) {return a+b};
var f_sub = function(a,b) {return a-b};
var f_mul = function(a,b) {return a*b};
var f_div = function(a,b) {return a/b};
var f_mod = function(a,b) {return a%b};

var f_and = function(a,b) {return a && b};
var f_or = function(a,b) {return a || b};
var f_not = function(a) {return !a};

var f_eq = function(a,b) {return a==b};
var f_neq = function(a,b) {return a!=b};
var f_peq = function(a,b) {return a===b};
var f_pneq = function(a,b) {return a!==b};
var f_gt = function(a,b) {return a>b};
var f_lt = function(a,b) {return a<b};
var f_geq = function(a,b) {return a>=b};
var f_leq = function(a,b) {return a<=b};

/** helpers **/
var overloader_2op = function(baseF, lifter1, lifter2) {
  return function(x1, x2) {
    if (isNumeric(x1) && isNumeric(x2))
      return baseF(x1, x2);
    else
      return lift_realreal_to_real(baseF, lifter1, lifter2, x1, x2);
  }
};
// this might need primal* if cmp operators are used with &rest
var overloader_2cmp = function(baseF) {
  var fn = function(x1, x2) {
      if (isDualNumber(x1) || isTape(x1))
        return fn(x1.primal, x2);
      else if (isDualNumber(x2) || isTape(x2))
        return fn(x1, x2.primal);
      else
        return baseF(x1, x2);
    }
  return function(x1, x2) {
    if (isNumeric(x1) && isNumeric(x2))
      return baseF(x1, x2);
    else {
      return fn(x1, x2);
    }
  };
};
var zeroF = function(x){return 0;};
var oneF = function(x1, x2){return 1.0;};
var m_oneF = function(x1, x2){return -1.0;};
var firstF = function(x1, x2){return x1;};
var secondF = function(x1, x2){return x2;};
var div2F = function(x1, x2){return d_div(1,x2);};
var divNF = function(x1, x2){return d_div(d_sub(0,x1), d_mul(x2, x2));};

/** lifted functions (overloaded) **/
var d_add = overloader_2op(f_add, oneF, oneF);
var d_sub = overloader_2op(f_sub, oneF, m_oneF);
var d_mul = overloader_2op(f_mul, secondF, firstF);
var d_div = overloader_2op(f_div, div2F, divNF);
// needswork: d_mod should be derived through `d_div` and `d_sub`
// needswork: logical and bitwise operations

var d_eq = overloader_2cmp(f_eq);
var d_neq = overloader_2cmp(f_neq);
var d_peq = overloader_2cmp(f_peq);
var d_pneq = overloader_2cmp(f_pneq);
var d_gt = overloader_2cmp(f_gt);
var d_lt = overloader_2cmp(f_lt);
var d_geq = overloader_2cmp(f_geq);
var d_leq = overloader_2cmp(f_leq);


var d_sqrt = function(x) {
  return lift_real_to_real(Math.sqrt, function(x){return d_div(1, d_mul(2.0, d_sqrt(x)))}, x)
};

var d_exp = function(x) {
  return lift_real_to_real(Math.exp, function(x){return d_exp(x)}, x);
};

var d_log = function(x) {
  return lift_real_to_real(Math.log, function(x){return d_div(1,x)}, x);
};

var d_floor = function(x) {
  return lift_real_to_real(Math.floor, zeroF, x);
};

var d_pow = function(x1, x2) {
  return lift_realreal_to_real(Math.pow,
                               function(x1, x2){return d_mul(x2, d_pow(x1, d_sub(x2, 1)));},
                               function(x1, x2){return d_mul(d_log(x1), d_pow(x1, x2));},
                               x1,
                               x2);
};

var d_sin = function(x) {
  return lift_real_to_real(Math.sin, function(x){return d_cos(x)}, x);
};

var d_cos = function(x) {
  return lift_real_to_real(Math.cos, function(x){return d_sub(0, d_sin(x))}, x);
};

var d_atan = function(x1, x2) {
  x2 = x2 === undefined ? 1 : x2; // just atan, not atan2
  return lift_realreal_to_real(Math.atan2,
                               function(x1, x2){return d_div(x2, d_add(d_mul(x1,x1), d_mul(x2,x2)));},
                               function(x1, x2){return d_div(d_sub(0,x1), d_add(d_mul(x1,x1), d_mul(x2,x2)));},
                               x1,
                               x2);
};


/** derivatives and gradients **/
var _e_ = 0

var lt_e = function(e1, e2) { return e1 < e2 }

var derivativeF = function(f) {
  return function(x) {
    _e_ += 1;
    var y = f(new S_dualNumber(_e_, x, 1.0));
    var y_prime = (!isDualNumber(y) || lt_e(y.epsilon, _e_)) ? 0.0 : y.perturbation;
    _e_ -= 1;
    return y_prime;
  }
}

var replace_ith = function(x, i, xi) { var c = x.slice(0); c[i] = xi; return c; }

var gradientF = function(f) {
  return function(x) {
    return x.map(function(xval) {
      var i = x.indexOf(xval);
      return derivativeF(function(xi) {return f(replace_ith(x, i, xi))})(xval);
    })
  }
}

var determineFanout = function(tape) {
  tape.fanout += 1;
  if (tape.fanout == 1) {
    var n = tape.tapes.length;
    while (n--) determineFanout(tape.tapes[n]);
  }
}

var reversePhase = function(sensitivity, tape) {
  tape.sensitivity = d_add(tape.sensitivity, sensitivity);
  tape.fanout -= 1;
  if (tape.fanout == 0) {
    var sens = tape.sensitivity;
    var n = tape.factors.length;
    while (n--) reversePhase(d_mul(sens, tape.factors[n]), tape.tapes[n]);
  }
}

var gradientR = function(f) {
  return function(x) {
    _e_ += 1;
    var new_x = x.map( function(xi) { return new S_tape(_e_, xi, [], []) } )
    var y = f(new_x);
    if (isTape(y) && !lt_e(y.epsilon, _e_)) {
      determineFanout(y);
      reversePhase(1.0, y);
    }
    _e_ -= 1;
    return new_x.map(function(v){return v.sensitivity})
  }
}

var derivativeR = function(f) {
  return function(x) {
    var r = gradientR( function(x1) {return f(x1[0])} )([x])
    return r[0]
  }
}

module.exports = {
  ad_add: d_add,
  ad_sub: d_sub,
  ad_mul: d_mul,
  ad_div: d_div,
  ad_eq: d_eq,
  ad_neq: d_neq,
  ad_peq: d_peq,
  ad_pneq: d_pneq,
  ad_gt: d_gt,
  ad_lt: d_lt,
  ad_geq: d_geq,
  ad_leq: d_leq,
  ad_floor: d_floor,
  ad_sqrt: d_sqrt,
  ad_exp: d_exp,
  ad_log: d_log,
  ad_pow: d_pow,
  ad_sin: d_sin,
  ad_cos: d_cos,
  ad_atan: d_atan,
  ad_derivativeF: derivativeF,
  ad_gradientF: gradientF,
  ad_derivativeR: derivativeR,
  ad_gradientR: gradientR,
};

// Also expose functions via the Math module
// TODO: ceil, tan, hyperbolic trig fns, abs, min(?), max(?)
var d_Math = {};
var mathProps = Object.getOwnPropertyNames(Math);
for (var i = 0; i < mathProps.length; i++) {
  var prop = mathProps[i];
  d_Math[prop] = Math[prop];
}
d_Math.floor = d_floor;
d_Math.sqrt = d_sqrt;
d_Math.exp = d_exp;
d_Math.log = d_log;
d_Math.pow = d_pow;
d_Math.sin = d_sin;
d_Math.cos = d_cos;
d_Math.atan = d_atan;
d_Math.atan2 = d_atan;
module.exports.ad_Math = d_Math;


