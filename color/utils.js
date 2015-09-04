
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

	// Norm of covariance matrix
	var covar = numeric.rep([15, 15], 0);
	for (j = 0; j < 15; j++) {
		for (k = 0; k < 15; k++) {
			for (var i = 0; i < labpalettes.length; i++) {
				covar[j][k] += (labpalettes[i][j] - mean[j])*(labpalettes[i][k] - mean[k]);
			}
			covar[j][k] /= (labpalettes.length - 1);
		}
	}
	var cvflat = _.flatten(covar);
	return Math.sqrt(numeric.sum(numeric.mul(cvflat, cvflat)) / cvflat.length);

	// // Sum of stddevs
	// var stddev = numeric.rep([15], 0);
	// for (var i = 0; i < labpalettes.length; i++) {
	// 	var diff = numeric.sub(labpalettes[i], mean);
	// 	numeric.muleq(diff, diff);
	// 	numeric.addeq(stddev, diff);
	// }
	// numeric.diveq(stddev, labpalettes.length - 1);
	// numeric.sqrteq(stddev);
	// return numeric.sum(stddev);
};

function loadTrainingData(ratingThreshold) {
	var ratings = fs.readFileSync(__dirname + '/data/mTurk_targets.txt').toString().split('\n').map(parseFloat);
	var palettes = fs.readFileSync(__dirname + '/data/mTurk_data.txt').toString().split('\n').map(function(d) {
		var nums = d.split(',').map(parseFloat);
		var pal = [];
		for (var i = 0; i < 5; i++)
			pal.push([nums[i], nums[5+i], nums[10+i]]);
		return pal;
	});
	var zipped = _.zip(palettes, ratings);
	var filtered = _.filter(zipped, function(x) { return x[1] >= ratingThreshold; });
	var unzipped = _.unzip(filtered);

	return {
		palettes: unzipped[0],
		ratings: unzipped[1]
	}
}
// // TEST
// var dat = loadTrainingData(3.7);
// saveStatsAndSamples('trainingData_3.7', dat.palettes, {trueRatings: dat.ratings});

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
		var filename;
		if (otherStats.trueRatings !== undefined)
			filename = dirname + '/' + rating + '_(' + otherStats.trueRatings[i] + ').png';
		else
			filename = dirname + '/' + rating + '.png';
		drawPalette(palettes[i], filename);
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
	loadTrainingData: loadTrainingData,
	saveStatsAndSamples: saveStatsAndSamples
};



