'use strict';

var fs = require('fs');
var sweet = require('sweet.js');


var macros = fs.readFileSync('./src/ad/macros.js');
function transform(code) {
	var allcode = macros + '\n' + code;
	var compiled = sweet.compile(allcode, {readableNames: true});
	return compiled.code;
}

module.exports = {
	transform: transform
}
