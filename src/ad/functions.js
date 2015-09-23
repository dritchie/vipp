"use strict";

function spaces(n) {
  var s = '';
  for (var i = 0; i < n; i++)
    s += ' ';
  return s;
}

var id = 0;
var S_tape = function(epsilon, primal) {
  this.epsilon = epsilon;
  this.primal = primal;
  this.fanout = 0;
  this.sensitivity = 0.0;
  this.opname = 'ROOT';
  this.id = id++;
};
S_tape.prototype = {
  determineFanout: function() { this.fanout += 1; },
  reversePhase: function(sensitivity) {
    this.sensitivity += sensitivity;
    this.fanout -= 1;
  },
  reversePhaseDebug: function(sensitivity, tablevel) {
    tablevel = tablevel === undefined ? 0 : tablevel;
    var oldsensitivity = this.sensitivity;
    this.sensitivity += sensitivity;
    this.fanout -=1;
    console.log(spaces(tablevel) + '[' + this.id + '] ' + this.opname + ' | primal: ' + this.primal + ', deriv: ' + oldsensitivity + ' + ' + sensitivity + ' = ' + this.sensitivity);
  },
  print: function(tablevel) {
    tablevel = tablevel === undefined ? 0 : tablevel;
    var s = spaces(tablevel);
    console.log(s + '[' + this.id + '] ' + this.opname + ' | primal: ' + this.primal + ', deriv: ' + this.sensitivity);
  },
  resetState: function() {
    this.sensitivity = 0.0;
    this.fanout = 0;
  },
  checkNaN: function() {
    for (var i = 0; i < arguments.length; i++) {
      if (isNaN(arguments[i])) {
        this.print();
        throw 'Found NaN in AD tape!';
      }
    }
  }
};
var isTape = function(t) { return t instanceof S_tape; };

var S_tape1 = function(opname, epsilon, primal, factor, tape) {
  S_tape.call(this, epsilon, primal);
  this.opname = opname;
  this.factor = factor;
  this.tape = tape;
}
S_tape1.prototype = new S_tape();
S_tape1.prototype.determineFanout = function() {
  this.fanout += 1;
  if (this.fanout === 1)
    this.tape.determineFanout();
}
S_tape1.prototype.reversePhase = function(sensitivity) {
  this.sensitivity += sensitivity;
  this.fanout -= 1;
  if (this.fanout === 0) {
    // ////
    // this.checkNaN(this.sensitivity*this.factor);
    // ////
    this.tape.reversePhase(this.sensitivity*this.factor);
  }
}
S_tape1.prototype.reversePhaseDebug = function(sensitivity, tablevel) {
  tablevel = tablevel === undefined ? 0 : tablevel;
  S_tape.prototype.reversePhaseDebug.call(this, sensitivity, tablevel);
  if (this.fanout === 0) {
    console.log(spaces(tablevel) + 'propagating. factor =  ' + this.factor + ', total: ' + this.sensitivity*this.factor);
    this.tape.reversePhaseDebug(this.sensitivity*this.factor, tablevel + 1);
  }
}
S_tape1.prototype.print = function(tablevel) {
  tablevel = tablevel === undefined ? 0 : tablevel;
  S_tape.prototype.print.call(this, tablevel);
  // console.log(spaces(tablevel) + 'factor: ' + this.factor);
  this.tape.print(tablevel + 1);
}
S_tape1.prototype.resetState = function() {
  S_tape.prototype.resetState.call(this);
  this.tape.resetState();
}

var S_tape2 = function(opname, epsilon, primal, factor1, factor2, tape1, tape2) {
  S_tape.call(this, epsilon, primal);
  this.opname = opname;
  this.factor1 = factor1;
  this.factor2 = factor2;
  this.tape1 = tape1;
  this.tape2 = tape2;
}
S_tape2.prototype = new S_tape();
S_tape2.prototype.determineFanout = function() {
  this.fanout += 1;
  if (this.fanout === 1) {
    this.tape1.determineFanout();
    this.tape2.determineFanout();
  }
}
S_tape2.prototype.reversePhase = function(sensitivity) {
  this.sensitivity += sensitivity;
  this.fanout -= 1;
  if (this.fanout === 0) {
    // ////
    // this.checkNaN(this.sensitivity*this.factor1, this.sensitivity*this.factor2);
    // ////
    this.tape1.reversePhase(this.sensitivity*this.factor1);
    this.tape2.reversePhase(this.sensitivity*this.factor2);
  }
}
S_tape2.prototype.reversePhaseDebug = function(sensitivity, tablevel) {
  tablevel = tablevel === undefined ? 0 : tablevel;
  S_tape.prototype.reversePhaseDebug.call(this, sensitivity, tablevel);
  if (this.fanout === 0) {
    // console.log(spaces(tablevel) + 'propagating. factor1 =  ' + this.factor1 + ', total: ' + this.sensitivity*this.factor1);
    // console.log(spaces(tablevel) + 'propagating. factor2 =  ' + this.factor2 + ', total: ' + this.sensitivity*this.factor2);
    this.tape1.reversePhaseDebug(this.sensitivity*this.factor1, tablevel + 1);
    this.tape2.reversePhaseDebug(this.sensitivity*this.factor2, tablevel + 1);
  }
}
S_tape2.prototype.print = function(tablevel) {
  tablevel = tablevel === undefined ? 0 : tablevel;
  S_tape.prototype.print.call(this, tablevel);
  console.log(spaces(tablevel) + 'factor1: ' + this.factor1 + ', factor2: ' + this.factor2);
  this.tape1.print(tablevel + 1);
  this.tape2.print(tablevel + 1);
}
S_tape2.prototype.resetState = function() {
  S_tape.prototype.resetState.call(this);
  this.tape1.resetState();
  this.tape2.resetState();
}

var lift_realreal_to_real = function(f, df_dx1, df_dx2) {
  var liftedfn;
  var fn = f;
  liftedfn = function(x_1, x_2) {
    if (isTape(x_1)) {
      if (isTape(x_2))
          return new S_tape2(f.name, x_1.epsilon,
                      fn(x_1.primal, x_2.primal),
                      df_dx1(x_1.primal, x_2.primal), df_dx2(x_1.primal, x_2.primal),
                      x_1, x_2)
      else
        return new S_tape1(f.name, x_1.epsilon, fn(x_1.primal, x_2), df_dx1(x_1.primal, x_2), x_1)
    }
    else {
      if (isTape(x_2))
        return new S_tape1(f.name, x_2.epsilon, fn(x_1, x_2.primal), df_dx2(x_1, x_2.primal), x_2)
      else
        return f(x_1, x_2)
    }
  };
  return liftedfn;
};

var lift_real_to_real = function(f, df_dx) {
  var liftedfn;
  var fn = f;
  liftedfn = function(x1) {
    if (isTape(x1))
      return new S_tape1(f.name, x1.epsilon, fn(x1.primal), df_dx(x1.primal), x1);
    else
      return f(x1);
  }
  return liftedfn;
};

/** functional wrappers for primitive operators **/
var f_minus = function f_minus(a) {return -a};

var f_add = function f_add(a,b) {return a+b};
var f_sub = function f_sub(a,b) {return a-b};
var f_mul = function f_mul(a,b) {return a*b};
var f_div = function f_div(a,b) {return a/b};
var f_mod = function f_mod(a,b) {return a%b};

var f_and = function f_and(a,b) {return a && b};
var f_or = function f_or(a,b) {return a || b};
var f_not = function f_not(a) {return !a};

var f_eq = function f_eq(a,b) {return a==b};
var f_neq = function f_neq(a,b) {return a!=b};
var f_peq = function f_peq(a,b) {return a===b};
var f_pneq = function f_pneq(a,b) {return a!==b};
var f_gt = function f_gt(a,b) {return a>b};
var f_lt = function f_lt(a,b) {return a<b};
var f_geq = function f_geq(a,b) {return a>=b};
var f_leq = function f_leq(a,b) {return a<=b};


var overloader_2cmp = function(baseF) {
  var fn = function(x1, x2) {
    if (isTape(x1)) {
      if (isTape(x2))
        return baseF(x1.primal, x2.primal);
      else
        return baseF(x1.primal, x2);
    } else if (isTape(x2))
      return baseF(x1, x2.primal);
    else
      return baseF(x1, x2);
  }
  return fn;
};

var zeroF = function(x){return 0;};
var oneF = function(x1, x2){return 1.0;};
var m_oneF = function(x1, x2){return -1.0;};
var firstF = function(x1, x2){return x1;};
var secondF = function(x1, x2){return x2;};
var div2F = function(x1, x2){return 1/x2;};
var divNF = function(x1, x2){return -x1/(x2*x2);};

/** lifted functions (overloaded) **/
var d_add = lift_realreal_to_real(f_add, oneF, oneF);
var d_sub = lift_realreal_to_real(f_sub, oneF, m_oneF);
var d_mul = lift_realreal_to_real(f_mul, secondF, firstF);
var d_div = lift_realreal_to_real(f_div, div2F, divNF);
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


var d_floor = lift_real_to_real(Math.floor, zeroF);
var d_ceil = lift_realreal_to_real(Math.ceil, zeroF);
var d_sqrt = lift_real_to_real(Math.sqrt, function(x){return 1/(2*Math.sqrt(x))});
var d_exp = lift_real_to_real(Math.exp, function(x){return Math.exp(x)});
var d_log = lift_real_to_real(Math.log, function(x){return 1/x});
var d_pow = lift_realreal_to_real(Math.pow,
                               function(x1, x2){return x2*Math.pow(x1, x2-1);},
                               function(x1, x2){return Math.log(x1)*Math.pow(x1, x2);});
var d_sin = lift_real_to_real(Math.sin, function(x){return Math.cos(x)});
var d_cos = lift_real_to_real(Math.cos, function(x){return -Math.sin(x)});
var d_atan_core = lift_realreal_to_real(Math.atan2,
                               function(x1, x2){return x2/(x1*x1 + x2*x2);},
                               function(x1, x2){return -x1/(x1*x1 + x2*x2);});
var d_atan = function(x1, x2) {
  x2 = x2 === undefined ? 1 : x2; // just atan, not atan2
  return d_atan_core(x1, x2);
};

var d_abs = function(x) {
  return d_lt(x, 0.0) ? d_sub(0.0, x) : x;
};

var d_min = function(x, y) {
  return d_lt(x, y) ? x : y;
};

var d_max = function(x, y) {
  return d_lt(x, y) ? y : x;
};

var d_tan = lift_real_to_real(Math.tan, function(x){
  var tx = Math.tan(x);
  return 1.0 + tx*tx;
});

var d_cosh = lift_real_to_real(Math.cosh, function(x){ Math.sinh(x); });

var d_sinh = lift_real_to_real(Math.sinh, function(x){ Math.cosh(x); });

var d_tanh = lift_real_to_real(Math.tanh, function(x){
  var tx = Math.tanh(x);
  return 1.0 - tx*tx;
});


/** derivatives and gradients **/
var _e_ = 0

var lt_e = function(e1, e2) { return e1 < e2 }

var gradientR = function(f) {
  return function(x) {
    _e_ += 1;
    var new_x = x.map( function(xi) { return new S_tape(_e_, xi) } )
    var y = f(new_x);
    if (isTape(y) && !lt_e(y.epsilon, _e_)) {
      y.determineFanout();
      y.reversePhase(1.0);
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
  ad_derivativeR: derivativeR,
  ad_gradientR: gradientR,
  ad_maketape: function(x) { return new S_tape(_e_, x); },
  ad_primal: function(x) { return x.primal === undefined ? x : x.primal; }
};

// Also expose functions via the Math module
var d_Math = {};
var mathProps = Object.getOwnPropertyNames(Math);
for (var i = 0; i < mathProps.length; i++) {
  var prop = mathProps[i];
  d_Math[prop] = Math[prop];
}
d_Math.floor = d_floor;
d_Math.ceil = d_ceil;
d_Math.sqrt = d_sqrt;
d_Math.exp = d_exp;
d_Math.log = d_log;
d_Math.pow = d_pow;
d_Math.sin = d_sin;
d_Math.cos = d_cos;
d_Math.atan = d_atan;
d_Math.atan2 = d_atan;
d_Math.abs = d_abs;
d_Math.min = d_min;
d_Math.max = d_max;
d_Math.tan = d_tan;
d_Math.sinh = d_sinh;
d_Math.cosh = d_cosh;
d_Math.tanh = d_tanh;
module.exports.ad_Math = d_Math;


