
var _ = require('underscore');
var fs = require('fs');


var RGB = {
	fromRGB: function(c) { return c.slice(); }
};


var HSV = {
	fromRGB: function(c) {
		var r = 255*c[0];
		var g = 255*c[1];
		var b = 255*c[2];
		var max = Math.max(r, Math.max(g, b));
		var min = Math.min(r, Math.min(g, b));
		var chroma = max - min;

		var huep = 0;
		if (chroma === 0) huep = 0; else
		if (max === r) huep = (g - b) / chroma % 6; else
		if (max === g) huep = (b - r) / chroma + 2; else
		if (max === b) huep = (r - g) / chroma + 4;

		var hue = 60 * huep;
		if (hue < 0)
			hue = hue + 360;
		var saturation = max === 0 ? 0 : 1 - min/max;
		var value = max / 255;

		return [hue, saturation, value];
	}
};


//sRGB to xyz using the D65 illuminant
//transformation from http://www.brucelindbloom.com
var M = [
	[0.4124564, 0.3575761, 0.1804375],
	[0.2126729, 0.7151522, 0.0721750],
	[0.0193339, 0.1191920, 0.9503041]
];
var LAB = {
	fromRGB: function(c) {
		var gamma = 2.2;
        var red = Math.pow(c[0], gamma);
        var green = Math.pow(c[1], gamma);
        var blue = Math.pow(c[2], gamma);

        var x = M[0][0] * red + M[0][1] * green + M[0][2] * blue;
        var y = M[1][0] * red + M[1][1] * green + M[1][2] * blue;
        var z = M[2][0] * red + M[2][1] * green + M[2][2] * blue;

        var XR = 0.95047;
        var YR = 1.00000;
        var ZR = 1.08883;

        var e = 216 / 24389.0;
        var k = 24389 / 27.0;

        var xR = x / XR;
        var yR = y / YR;
        var zR = z / ZR;

        var fx = xR > e ? Math.pow(xR, 1 / 3.0) : (k * xR + 16) / 116.0;
        var fy = yR > e ? Math.pow(yR, 1 / 3.0) : (k * yR + 16) / 116.0;
        var fz = zR > e ? Math.pow(zR, 1 / 3.0) : (k * zR + 16) / 116.0;

        var cieL = 116 * fy - 16;
        var cieA = 500 * (fx - fy);
        var cieB = 200 * (fy - fz);

        return [cieL, cieA, cieB];
	}
};


function PiecewisePolynomial(filename) {
	// First line are the breaks
	// Next lines are the coefficients
	var lines = fs.readFileSync(filename).toString().split('\n');
	if (lines[lines.length-1] === '')
		lines.splice(lines.length-1, 1);
	var order = parseInt(lines[0]);
	var breaks = lines[1].split(',').map(parseFloat);
	lines = lines.slice(2);
	var coefficients = lines.map(function(line) {
		return line.split(',').map(parseFloat);
	});

	this.breaks = breaks;
	this.coefficients = coefficients;
	this.order = order;
};
PiecewisePolynomial.prototype = {
	constructor: PiecewisePolynomial,
	evalAt: function(x) {
		var idx = _.sortedIndex(this.breaks, x) - 1;
		var delta = x - this.breaks[idx];
		var result = 0;
		for (var i = 0; i <= this.order; i++)
			result += this.coefficients[idx][i] * Math.pow(delta, this.order - i);
		return result;
	}
};


var hueRemap = new PiecewisePolynomial(__dirname + '/data/hueRemap.txt');
function deg2rad(x) { return x * Math.PI / 180; }
function rad2deg(x) { return x * 180 / Math.PI; }
var CHSV = {
	fromRGB: function(c) {
		var hsv = HSV.fromRGB(c);
		var remap = deg2rad(360*hueRemap.evalAt(hsv[0]/360));
		return [hsv[1]*Math.cos(remap), -hsv[1]*Math.sin(remap), hsv[2]];
	}
};

module.exports = {
	RGB: RGB,
	HSV: HSV,
	LAB: LAB,
	CHSV: CHSV
};


