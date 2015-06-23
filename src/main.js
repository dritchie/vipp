
'use strict';


var adtransform = require('src/ad/transform.js').transform;


function compile(code) {
	// AD and eval the code to get a callable thunk
	var adcode = adtransform(code);
	var wrappedcode = '(function() {\n' + adcode + '\n})\n';
	var fn = eval(wrappedcode);
	return function() {
		// Install header + AD stuff into the global environment.
		var oldG = {};
		var ad = require('src/ad/functions');
		for (var prop in ad) {
			oldG[prop] = global[prop];
			global[prop] = ad[prop];
		}
		var header = require('src/header');
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