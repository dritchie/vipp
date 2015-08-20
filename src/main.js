
'use strict';


var adtransform = require('./ad/transform.js').transform;
var namingtransform = require('./naming/naming.js').transform;


function compile(code) {
	// Apply source transforms
	code = '(function() {\n' + code + '\n})\n';
	code = namingtransform(code);
	// require('fs').writeFileSync('test.js', code);
	code = adtransform(code);
	// Eval the code to get a callable thunk
	var fn = eval(code);
	var thunk = function() {
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
	// Return the thunk and the code
	return {
		fn: thunk,
		code: code
	};
}


module.exports = {
	compile: compile
};