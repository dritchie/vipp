
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
var hugeAdj = loadMatrix(__dirname + '/' + haFile);
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

function getAllFeatures(palette) {
	//
}

function getRating(palette) {
	var features = getAllFeatures(palette);
	assert(features.length === weights.length,
		'features and weights have different length!');
	features.push(1);  // constant bias term
	var score = 0;
	for (var i = 0; i < features.length; i++)
		score += weights[i]*features[i];
	return score;
}


module.exports = {
	getRating: getRating
};




