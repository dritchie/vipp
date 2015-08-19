'use strict';

var Syntax = require('estraverse').Syntax;
var replace = require('estraverse').replace;
var build = require('ast-types').builders;

var makeGensym = require('./util').makeGensym;
var makeGenvar = require('./syntax').makeGenvar;
var inProgram = require('./syntax').inProgram;
var fail = require('./syntax').fail;
var isPrimitive = require('./syntax').isPrimitive;

var esprima = require('esprima');
var escodegen = require('escodegen');


// Functions we should avoid transforming
var blacklist = ['require'];
var tmp = {};
for (var i = 0; i < blacklist.length; i++)
  tmp[blacklist[i]] = true;
blacklist = tmp;


function makeGenlit() {
  var gensym = makeGensym();
  return function() {
    return build.literal(gensym('_'));
  };
}

var genlit = null;
var genvar = null;

var addresses = [];

function makeAddressExtension(address) {
  return build.callExpression(
      build.memberExpression(address,
                             build.identifier('concat'),
                             false),
      [genlit()]);
}

function generating(node) {
  switch (node.type) {
    case Syntax.FunctionExpression:
      addresses.unshift(genvar('address'));
      break;
    default:
  }
}

function naming(node) {
  switch (node.type) {
    case Syntax.FunctionExpression:
      return build.functionExpression(node.id,
          [addresses.shift()].concat(node.params),
          node.body);

    // add a gensym onto the address variable
    case Syntax.CallExpression:
      if (isPrimitive(node.callee) || blacklist[node.callee.name]) {
        return node;
      } else {
        return build.callExpression(node.callee,
            [makeAddressExtension(addresses[0])].concat(node.arguments));
      }

    default:
  }
}

function namingMain(node) {
  genlit = makeGenlit();
  genvar = makeGenvar();

  return replace(node, {enter: generating, leave: naming});
}

function transform(code) {
  var ast = esprima.parse(code);
  ast = namingMain(ast);
  var out = escodegen.generate(ast);
  return out;
}


// // TEST
// var fs = require('fs');
// var code = fs.readFileSync('./procmod/programs/spaceship.js');
// code = transform(code);
// fs.writeFileSync('./test.js', code);


module.exports = {
  transform: transform
};
