
var vipp = require('../../src/main');
var variational = require('../../src/variational.js')
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
//       enforceConvergence: whether to bail if inference fails to converge (defaults to true)
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
	fs.writeSync(overallFile, 'condition,steps,time,elbo,eubo\n');
	fs.writeSync(stepFile, 'condition,step,time,elbo,eubo\n');
	// Run
	var enforceConvergence = options.enforceConvergence === undefined ? true : options.enforceConvergence;
	for (var i = 0; i < conditions.length; i++) {
		var condition = conditions[i];
		for (var j = 0; j < options.numRuns; j++) {
			process.stdout.write(' Running condition "' + condition.name + '" (Run ' + (j+1) + '/' + options.numRuns + ')\r');
			var results = runCondition(code, condition);
			if (enforceConvergence) assert(results.converged);
			// Write data to .csv
			fs.writeSync(overallFile,
				[condition.name,
				 results.stepsTaken,
				 results.timeTaken,
				 results.elbo,
				 results.eubo].toString()+'\n');
			var time = results.stepStats.time;
			var elbo = results.stepStats.elbo;
			var eubo = results.stepStats.eubo;
			for (var k = 0; k < time.length; k++) {
				fs.writeSync(stepFile, [condition.name, k, time[k], elbo[k], eubo[k]].toString()+'\n');
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


// Simpler version: takes a filename + all the same args that infer takes
function run(filename, target, guide, args, opts) {
	opts.recordStepStats = true;
	var results = variational.infer(target, guide, args, opts);
	var file = fs.openSync(filename, 'w');
	fs.writeSync(file, 'step,time,elbo,eubo\n');
	var time = results.stepStats.time;
	var elbo = results.stepStats.elbo;
	var eubo = results.stepStats.eubo;
	for (var k = 0; k < time.length; k++) {
		fs.writeSync(file, [k, time[k], elbo[k], eubo[k]].toString()+'\n');
	}
	return results;
}


module.exports = {
	runExperiment: runExperiment,
	run: run
}



