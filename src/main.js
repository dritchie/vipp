
'use strict';

var syscall = require('child_process').execSync;
var fs = require('fs');


// This currently uses sweetify do to all the macro expansion
// This won't work in the browwer.
// TODO: Have a makefile-based compilation system that compiles
//    erp, variational, etc. into one browser-loadable blob,
//    then for the actual program code, use the raw sweet.js API
//    to transform it (no filesystem access needed).
var outfile = '_vipp_out.js'
var supportFiles = [
	'./src/variational.sjs',
	'./src/erp.sjs',
	'./src/util.sjs'
];
function executeFile(filename) {
	// Call sweetify on this file and a bunch of supporting files
	var cmd = 'browserify -t sweetify ';
	for (var i = 0; i < supportFiles.length; i++)
		cmd += supportFiles[i] + ' ';
	cmd += filename + ' > ' + outfile;
	syscall(cmd);

	// Load that code and eval it.
	var code = fs.readFileSync(outfile);
	try {
		return eval(code);
	} finally {
		// Clean up
		syscall('rm -f ' + outfile);
	}
};

var infile = '_vipp_in.sjs';
function executeCode(code) {
	// Save code to file, then call 'compile(filename)'
	fs.writeFileSync(infile, code);
	try {
		return executeFile(infile);
	} finally {
		// Clean up
		syscall('rm -f ' + infile);
	}
}


module.exports = {
	executeCode: executeCode,
	executeFile: executeFile
};