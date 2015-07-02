
var vipp = require('../../src/main');
var adtransform = require('../../src/ad/transform.js').transform;
var fs = require('fs');
var syscall = require('child_process').execSync;
var assert = require('assert');

// modelFileName: .js file where the model lives. Assumed to contain one
//    function called 'target' and one or more guide functions (named however)
// conditions: The different experimental conditions. This is an object with
//    the following structure:
//    {
//       name: Just a readable name.
//	     guide: Name of the guide function to use.
//       vars: Any global variables that should be defined (a js object).
//       opts: Any opts that should be passed to infer (a js object).
//    }
// options: options controlling experiment behavior. This is an object with:
//    {
//       outBaseName: Base filename for .csv file output. Defaults to modelFileName.
//	     numRuns: Number of runs of each condition to record.
//       modelArgs: Any input arguments that the target/guide need (i.e. evidence/data).
//          Defaults to undefined.
//    }
function runExperiment(modelFileName, conditions, options) {
	// Invoke make, to be sure that everything is up-to-date
	syscall('make', {stdio: null});
	// Load the code, append the inference stuff to it, do AD transform
	// (We do the AD transform once up front so we don't have to do it over and over
	//    again for every condition/run).
	var code = fs.readFileSync(modelFileName);
	code = adtransform(code);
	// Set up
	var old__args = global.__args;
	global.__args = options.modelArgs;
	var basename = options.outBaseName || modelFileName;
	var overallFile = fs.openSync(basename + '_overall.csv', 'w');
	var stepFile = fs.openSync(basename + '_step.csv', 'w');
	fs.writeSync(overallFile, 'condition,steps,time,elbo\n');
	fs.writeSync(stepFile, 'condition,step,time,elbo\n');
	// Run
	for (var i = 0; i < conditions.length; i++) {
		var condition = conditions[i];
		for (var j = 0; j < options.numRuns; j++) {
			process.stdout.write(' Running condition "' + condition.name + '" (Run ' + (j+1) + '/' + options.numRuns + ')\r');
			var results = runCondition(code, condition);
			assert(results.converged);
			// Write data to .csv
			fs.writeSync(overallFile,
				[condition.name,
				 results.stepsTaken,
				 results.timeTaken,
				 results.elbo].toString()+'\n');
			var time = results.stepStats.time;
			var elbo = results.stepStats.elbo;
			assert(time.length === elbo.length);
			for (var k = 0; k < time.length; k++) {
				fs.writeSync(stepFile, [condition.name, k, time[k], elbo[k]].toString()+'\n');
			}
		}
		process.stdout.write('\n');
	}
	// Clean up
	global.__args = old__args;
	fs.closeSync(overallFile);
	if (stepFile !== null)
		fs.closeSync(stepFile);
}

// Run a thunk until it doesn't throw an exception.
function untilSuccess(fn) {
	while (true) {
		try {
			return fn();
		} catch (e) {}
	}
}

function runCondition(code, condition) {
	// Set up
	var old__options = global.__options;
	global.__options = condition.opts;
	condition.opts.recordStepStats = true;
	var vars = condition.vars || {};
	var oldVars = {};
	for (var vname in vars) {
		oldVars[vname] = global[vname];
		global[vname] = vars[vname];
	}
	// Run
	var suffix = '\nreturn infer(target,' + condition.guide + ', __args, __options);\n';
	code = code + suffix;
	var fn = vipp.compile(code, false);
	var ret = untilSuccess(fn);
	// Clean up
	global.__options = old__options;
	for (var vname in oldVars) {
		global[vname] = oldVars[vname];
	}
	return ret;
}


module.exports = {
	runExperiment: runExperiment
}



