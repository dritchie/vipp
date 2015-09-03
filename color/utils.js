
var colorCompat = require('colorcompatibility');
var colorSpaces = require('colorcompatibility/colorSpaces');
var numeric = require('numeric');
var fs = require('fs');
var cp = require('child_process');
var _ = require('underscore');

function drawPalette(palette, filename) {
	var width = 100;
	var height = 200;

	var Canvas = require('canvas')
	  , canvas = new Canvas(width*palette.length, height)
	  , ctx = canvas.getContext('2d')
	  , fs = require('fs');

	for (var i = 0; i < palette.length; i++) {
		var color = palette[i];
		ctx.fillStyle = 'rgb(' +
			Math.floor(color[0]*255) + ',' + 
			Math.floor(color[1]*255) + ',' + 
			Math.floor(color[2]*255) + ')';
		ctx.fillRect(i*width, 0, width, height);
	}

	fs.writeFileSync(filename, canvas.toBuffer());
}

function computeDiversity(palettes) {
	// Do this in LAB space, since that's the most perceptual color space we have
	// Also flatten out the palettes so we just have one big array
	var labpalettes = palettes.map(function(p) {
		return _.flatten(p.map(colorSpaces.LAB.fromRGB));
	});
	var mean = numeric.rep([15], 0);
	for (var i = 0; i < labpalettes.length; i++)
		numeric.addeq(mean, labpalettes[i]);
	numeric.diveq(mean, labpalettes.length);
	var stddev = numeric.rep([15], 0);
	for (var i = 0; i < labpalettes.length; i++) {
		var diff = numeric.sub(labpalettes[i], mean);
		numeric.muleq(diff, diff);
		numeric.addeq(stddev, diff);
	}
	numeric.diveq(stddev, labpalettes.length);
	numeric.sqrteq(stddev);
	return numeric.sum(stddev);
};

// TODO: Switch everything (including webppl version) to use this
function saveStatsAndSamples(name, palettes, otherStats) {
	if (otherStats === undefined) otherStats = {};
	var dirname = 'color/results/' + name;
	if (fs.existsSync(dirname))
		cp.execSync('rm -rf ' + dirname);
	fs.mkdirSync(dirname);
	var avgRating = 0;
	for (var i = 0; i < palettes.length; i++) {
		var rating = colorCompat.getRating(palettes[i]);
		avgRating += rating;
		drawPalette(palettes[i], dirname + '/' + rating + '.png');
	}
	avgRating /= palettes.length;
	var diversity = computeDiversity(palettes);
	var stats = _.extend({
		averageRating: avgRating,
		diversity: diversity
	}, otherStats);
	fs.writeFileSync(dirname + '/stats.txt', JSON.stringify(stats));
}

module.exports = {
	saveStatsAndSamples: saveStatsAndSamples
};