'use strict';

var erp = require('src/erp.js');
var variational = require('src/variational.js');

module.exports = {};

for (var prop in erp)
	module.exports[prop] = erp[prop];
for (var prop in variational)
	module.exports[prop] = variational[prop];	