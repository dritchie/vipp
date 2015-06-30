
// Must be run from the repo root.

var vipp = require('../../src/main');
var adtransform = require('../../src/ad/transform.js').transform;
var fs = require('fs');

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
//       recordStepStats: Whether to record step-by-step convergence statistics
//          and save them in 'outBaseName_step.csv'.
//    }
function runExperiment(modelFileName, conditions, options) {
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
	var stepFile = options.recordStepStats ? fs.openSync(basename + '_step.csv', 'w') : null;
	fs.writeSync(overallFile, 'condition,steps,time,elbo\n');
	if (stepFile !== null)
		fs.writeSync(stepFile, 'condition,step,time,elbo\n');
	// Run
	for (var i = 0; i < conditions.length; i++) {
		var condition = conditions[i];
		var results = runCondition(code, condition);
		// Write data to .csv
		// TODO: Finish this
	}
	// Clean up
	global.__args = old__args;
	fs.closeSync(overallFile);
	if (stepFile !== null)
		fs.closeSync(stepFile);
}

function runCondition(code, condition) {
	// Set up
	var old__options = global.__options;
	global.__options = condition.opts;
	var oldVars = {};
	for (var vname in condition.vars) {
		oldVars[vname] = global[vname];
		global[vname] = condition.vars[vname];
	}
	// Run
	var suffix = '\nreturn infer(target,' + condition.guide + ', __args, __options);\n';
	code = code + suffix;
	var fn = vipp.compile(code, false);
	var ret = fn();
	// Clean up
	global.__options = old__options;
	for (var vname in oldVars) {
		global[vname] = oldVars[vname];
	}
	return ret;
}