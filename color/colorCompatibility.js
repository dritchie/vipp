
var _ = require('underscore');
var fs = require('fs');
var numeric = require('numeric');
var assert = require('assert');
var ColorSpaces = require('./colorSpaces');


var weightsFile = 'data/compatWeights.txt';
var hjFile = 'data/hueJoint.txt';
var haFile = 'data/hueAdj.txt';
var hpFile = 'data/hueProb.txt';
var weights = loadVector(__dirname + '/' + weightsFile);
var hueJoint = loadMatrix(__dirname + '/' + hjFile);
var hueAdj = loadMatrix(__dirname + '/' + haFile);
var hueProb = loadVector(__dirname + '/' + hpFile);


function loadVector(filename) {
	var lines = fs.readFileSync(filename).toString().split('\n');
	if (lines[lines.length-1] === '')
		lines.splice(lines.length-1, 1);
	return lines.map(parseFloat);
}

function loadMatrix(filename) {
	var lines = fs.readFileSync(filename).toString().split('\n');
	if (lines[lines.length-1] === '')
		lines.splice(lines.length-1, 1);
	return lines.map(function(line) {
		return line.split(',').map(parseFloat);
	});
}

// ----------------------------------------------------------------------------

// concat a bunch of vectors
function vconcat() {
	var result = [];
	for (var i = 0; i < arguments.length; i++) {
		for (var j = 0; j < arguments[i].length; j++)
			result.push(arguments[i][j]);
	}
	return result;
}

// like concat, but adds to the first argument in place
function vadd() {
	var result = arguments[0];
	for (var i = 1; i < arguments.length; i++) {
		for (var j = 0; j < arguments[i].length; j++)
			result.push(arguments[i][j]);
	}
}

// ----------------------------------------------------------------------------

// Some vector operations that we need in multiple places
function vmin(x) {
	var min = Number.MAX_VALUE;
	for (var i = 0; i < x.length; i++)
		min = Math.min(min, x[i]);
	return min;
}
function vmax(x) {
	var max = -Number.MAX_VALUE;
	for (var i = 0; i < x.length; i++)
		max = Math.max(max, x[i]);
	return max;
}
function vmean(x) {
	return numeric.sum(x) / x.length;
}
function vstddev(x, mean) {
	if (mean === undefined) mean = vmean(x);
	var stddev = 0;
	for (var i = 0; i < x.length; i++) {
		var diff = x[i] - mean;
		stddev += diff*diff;
	}
	return Math.sqrt(stddev / x.length);
}

// ----------------------------------------------------------------------------

// Hue probability stuff

var veps = numeric.rep([8], 1e-6);
var check = function(d) { return isFinite(d) ? d : 0; };
function getBasicStats(x) {
	if (x.length === 0)
		return numeric.rep([8], 0);
	var logx = numeric.log(numeric.add(x, veps));
	var meanx = vmean(x);
	var meanlogx = vmean(logx);
	return [
		check(meanx),
		check(vstddev(x, meanx)),
		check(vmin(x)),
		check(vmax(x)),
		check(meanlogx),
		check(vstddev(x, meanlogx)),
		check(vmin(logx)),
		check(vmax(logx))
	];
}


function circVMPdf(alpha, thetahat) {
	//kappa set to 2*pi
    //besseli(0, 2*pi) = 87.1085
    var kappa = 2*Math.PI;
    var besseli = 87.1085;
    var C = 1 / (2*Math.PI*besseli);
    var p = numeric.sub(alpha, thetahat);
    numeric.coseq(p);
    numeric.muleq(p, kappa);
    numeric.expeq(p);
    numeric.muleq(p, C);
    return p;
}

function getHueProbs(palette, satThresh) {
	if (satThresh === undefined) satThresh = 0.2;

	var validHues = palette.filter(function(c) { return c[1] > satThresh; })
						   .map(function(c) { return Math.round(c[0]*359) + 1; });
	var hueJointList = numeric.rep([validHues.length * (validHues.length+1) / 2], 0);
	var idx = 0;
	for (var i = 0; i < validHues.length; i++) {
		for (var j = i; j < validHues.length; j++) {
			hueJointList[idx] = hueJoint[validHues[i]-1][validHues[j]-1];
			idx++;
		}
	}

	var hueAdjList = [];
	for (var i = 1; i < validHues.length; i++)
		hueAdjList.push(hueAdj[validHues[i-1]][validHues[i]]);
	var hueProbList = validHues.map(function(h) { return hueProb[h-1]; });

	var entropy;
	if (validHues.length > 0) {
		var alpha = numeric.linspace(0, 2*Math.PI, 361).slice(0, 360);
		var pMix = _.foldl(validHues, function(b, a) {
			return numeric.add(b, circVMPdf(alpha, a*2*Math.PI));
		}, numeric.rep([alpha.length], 0.001));
		var normPMix = numeric.diveq(pMix, numeric.sum(pMix));
		var logPMix = numeric.log(normPMix);
		entropy = -numeric.sum(numeric.mul(normPMix, logPMix));
	} else entropy = 5.9;	// Set entropy high

	var hpf = getBasicStats(hueProbList);
	var hjf = getBasicStats(hueJointList);
	var haf = getBasicStats(hueAdjList);

	return vconcat(hpf, hjf, haf, [entropy]);
}

// ----------------------------------------------------------------------------

// Plane features stuff

function pca2(matrix) {
	var Y = numeric.diveq(matrix, Math.sqrt(matrix.length-1));
	var SVD = numeric.svd(Y);
	var s = SVD.S; var p = SVD.V;

	// variances
	var v = numeric.mul(s, s);

	return {coeff: p, roots: v};
}

function getPlaneFeatures(matrixOrig) {
	var matrix = numeric.clone(matrixOrig);
	var means = numeric.transpose(matrix).map(vmean);

	// subtract means
	for (var i = 0; i < matrix.length; i++)
		numeric.subeq(matrix[i], means);

	var PCA = pca2(matrix);
	var normal = numeric.transpose(PCA.coeff)[2];
	if (normal[0] < 0)
		numeric.muleq(normal, -1);

	var sumroots = numeric.sum(PCA.roots);
	var pctExplained = sumroots === 0 ? numeric.rep([3], 0) : numeric.div(PCA.roots, sumroots);
	var error = numeric.dot(matrix, normal).map(Math.abs);
	numeric.muleq(error, error);
	var sse = numeric.sum(error);

	return vconcat(normal, pctExplained, [sse]);
}

// ----------------------------------------------------------------------------

// 'Main' stuff

var colorSpaces = [];
for (var name in ColorSpaces) colorSpaces.push(ColorSpaces[name]);
function getAllFeatures(palette) {
	var allFeatures = [];
	for (var cs = 0; cs < colorSpaces.length; cs++) {
		var cspace = colorSpaces[cs];

		var convertedPalette;
		// Need extra gamma correction if LAB
		if (cspace !== ColorSpaces.LAB) {
			convertedPalette = palette.map(cspace.fromRGB);
		} else {
			var gamma = 2.2;
			convertedPalette = palette.map(function(c) {
				return cspace.fromRGB(c.map(function(x) { return Math.pow(x, 1/gamma); }));
			});
		}

		// Normalize
		if (cspace === ColorSpaces.HSV) {
			for (var c = 0; c < convertedPalette.length; c++)
				convertedPalette[c][0] /= 360;
		} else if (cspace === ColorSpaces.LAB) {
			for (var c = 0; c < convertedPalette.length; c++) {
				convertedPalette[c][0] /= 100;
				convertedPalette[c][1] /= 128;
				convertedPalette[c][2] /= 128;
			}
		}

		// Color coordinates
		var coords = _.flatten(convertedPalette);

		// Colors sorted by 3rd dimension
		var sortedCoords = _.flatten(convertedPalette.slice().sort(function(c1,c2) { return c1[2] - c2[2]; }));

		// Pairwise channel differences
		var pairwiseDiffs = function(palette, c) {
			var diffs = [];
			for (var i = 1; i < palette.length; i++) {
				diffs.push(palette[i][c] - palette[i-1][c]);
			}
			return diffs;
		};
		var diffs1 = pairwiseDiffs(convertedPalette, 0);
		if (cspace === ColorSpaces.HSV) {
			for (var i = 0; i < diffs1.length; i++) {
				var hdiff = Math.abs(diffs1[i]);
				diffs1[i] = Math.min(hdiff, 1-hdiff);
			}
		}
		var diffs2 = pairwiseDiffs(convertedPalette, 1);
		var diffs3 = pairwiseDiffs(convertedPalette, 2);

		// Pairwise sorted channel differences
		var sorteddiffs1 = diffs1.slice().sort(function(d1,d2) { return d2 - d1; });
		var sorteddiffs2 = diffs2.slice().sort(function(d1,d2) { return d2 - d1; });
		var sorteddiffs3 = diffs3.slice().sort(function(d1,d2) { return d2 - d1; });

		// Mean, stddev, min, max, median of each channel
		var channels = numeric.transpose(convertedPalette);
		var means = channels.map(vmean);
		var stddevs = channels.map(vstddev);
		var mins = channels.map(vmin);
		var maxs = channels.map(vmax);
		var medians = [];
		for (var i = 0; i < channels.length; i++) {
			channels[i].sort();
			medians.push(channels[i][Math.floor(channels.length/2)]);
		}

		// max-min difference for each channel
		var maxMin = numeric.sub(maxs, mins);

		// if HSV: hue probability features
		// otherwise: plane features
		var extraFeatures = cspace === ColorSpaces.HSV ? getHueProbs(convertedPalette) : getPlaneFeatures(convertedPalette);

		// Add new features
		vadd(allFeatures,
			coords, sortedCoords, diffs1, diffs2, diffs3, sorteddiffs1, sorteddiffs2, sorteddiffs3,
			means, stddevs, medians, maxs, mins, maxMin, extraFeatures);
	}

	return allFeatures;
}

function getRating(palette) {
	var features = getAllFeatures(palette);
	features.push(1);  // constant bias term
	// console.log(features.length, weights.length);
	assert(features.length === weights.length,
		'features and weights have different length!');
	var score = 0;
	for (var i = 0; i < features.length; i++)
		score += weights[i]*features[i];
	return score;
}


// TEST
var testPalette = [
	[0.1176, 0.5373, 0.1294],
	[0.0863, 0.2745, 0.6824],
	[0.9176, 0.7922, 0.0314],
	[0.6902, 0.1098, 0.0314],
	[0.0863, 0.2745, 0.6824]
];
console.log(getRating(testPalette));
// should be: 2.5729


module.exports = {
	getRating: getRating
};




