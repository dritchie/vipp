// NOTE: This script must be run from the repo root repository
// i.e. 'node experiments/adTest.js'

var ad = require('../../src/ad/functions.js');
var adtransform = require('../../src/ad/transform.js').transform;
var fs = require('fs');

// Install the AD functions globally
for (var prop in ad) {
	global[prop] = ad[prop];
}

// Set up the two versions of the code: ADed and non-ADed
var rawcode = fs.readFileSync('./experiments/adTest/adTestModel.js').toString();
var makeRunTest = eval(rawcode)();
var adcode = adtransform(rawcode);
var makeRunTestAD = eval(adcode)();

function hrtimeToSeconds(t) {
	// Seconds + nanoseconds
	return t[0] + t[1]/1e9;
}

// Run tests at various data sizes and conditions
var conditions = [
	'noad',
	'adPrimal',
	'adDual',
	'adGradient'
];
var sizes = [];
for (var i = 100; i <= 1000; i += 100 ) sizes.push(i);
var numRuns = 40;
var mu0 = 0.5;
var sigma0 = 1.5;
var csvFile = fs.openSync('./experiments/adTest/adTest_js.csv', 'w');
fs.writeSync(csvFile, 'condition,size,time\n');
for (var c = 0; c < conditions.length; c++) {
	var condition = conditions[c];
	console.log('CONDITION = ' + condition);
	var makeTest = (condition === 'noad' ? makeRunTest : makeRunTestAD);
	var useDualInputs = (condition === 'adDual');
	for (var s = 0; s < sizes.length; s++) {
		var size = sizes[s];
		console.log('  size = ' + size);
		var mu = useDualInputs ? new ad.tape(0, mu0, [], []) : mu0;
		var sigma = useDualInputs ? new ad.tape(0, sigma0, [], []) : sigma0;
		var testfn = makeTest(size);
		if (condition === 'adGradient') {
			var f = testfn;
			testfn = function(mu, sigma) {
				return ad.ad_gradientR(function(params) {
					return f(params[0], params[1]);
				})([mu, sigma]);
			};
		}
		// Actually run
		for (var r = 0; r < numRuns; r++) {
			var t0 = process.hrtime();
			testfn(mu, sigma);
			var tdiff = process.hrtime(t0);
			var timediff = hrtimeToSeconds(tdiff);
			fs.writeSync(csvFile, [condition, size, timediff].toString() + '\n');
		}
	}
}
fs.closeSync(csvFile);