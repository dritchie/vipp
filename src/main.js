
'use strict';


var adtransform = require('./ad/transform.js').transform;
var namingtransform = require('./naming/naming.js').transform;


function compile(code) {
	// Apply source transforms
	code = '(function() {\n' + code + '\n})\n';
	code = namingtransform(code);
	code = adtransform(code);
	// Eval the code to get a callable thunk
	var fn = eval(code);
	return function() {
		// Install header + AD stuff into the global environment.
		var oldG = {};
		var ad = require('./ad/functions');
		for (var prop in ad) {
			oldG[prop] = global[prop];
			global[prop] = ad[prop];
		}
		var header = require('./header');
		for (var prop in header) {
			oldG[prop] = global[prop];
			global[prop] = header[prop];
		}
		// Run the code.
		var ret = fn('');
		// Restore the global environment.
		for (var prop in oldG)
			global[prop] = oldG[prop];
		// Return
		return ret;
	}
}


module.exports = {
	compile: compile
};