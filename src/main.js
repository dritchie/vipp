
'use strict';


var adtransform = require('./ad/transform.js').transform;


function compile(code, doADtransform) {
	doADtransform = doADtransform === undefined ? true : doADtransform;
	// AD and eval the code to get a callable thunk
	if (doADtransform) code = adtransform(code);
	// Eval the code to get a callable thunk
	var wrappedcode = '(function() {\n' + code + '\n})\n';
	var fn = eval(wrappedcode);
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
		var ret = fn();
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