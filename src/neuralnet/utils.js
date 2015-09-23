var assert = require('assert');
var numeric = require('numeric');

// Create a neural net input sample cache
function makeInputSampleCache(nSamples) {
	var c = {
		nToCollect: nSamples,
		cache: {}, 
	};
	// This assumes that we've hit every relevant
	//    callsite at least once...
	c.hasEnoughSamples = function() {
		for (var callsite in c.cache) {
			if (c.cache[callsite].n < c.nToCollect)
				return false;
		}
		return true;
	};
	return c;
}

// Collect an input sample for a given callsite
function collectSample(callsite, inputs, inputCache) {
	var cacheEntry = inputCache.cache[callsite];
	if (cacheEntry === undefined) {
		cacheEntry = {
			n: 0,
			// mins: numeric.rep([inputs.length], Number.MAX_VALUE),
			// maxs: numeric.rep([inputs.length], -Number.MAX_VALUE)
			means: numeric.rep([inputs.length], 0),
			variances: numeric.rep([inputs.length], 0)
		};
		inputCache.cache[callsite] = cacheEntry;
	}
	if (cacheEntry.n < inputCache.nToCollect) {
		cacheEntry.n++;
		// numeric.mineq(cacheEntry.mins, inputs);
		// numeric.maxeq(cacheEntry.maxs, inputs);
		var prevmeans = cacheEntry.means;
		cacheEntry.means = numeric.add(prevmeans, numeric.div( numeric.sub(inputs, prevmeans), cacheEntry.n ));
		var prevvar = cacheEntry.variances;
		cacheEntry.variances = numeric.add(prevvar, numeric.mul( numeric.sub(inputs, prevmeans), numeric.sub(inputs, cacheEntry.means) ));
		if (cacheEntry.n === inputCache.nToCollect) {
			// cacheEntry.ranges = numeric.sub(cacheEntry.maxs, cacheEntry.mins);
			numeric.diveq(cacheEntry.variances, cacheEntry.n - 1);
			cacheEntry.stddevs = numeric.sqrt(cacheEntry.variances);
		}
	}
}

// Normalize NN inputs given collected stats
function normalizeInputs(callsite, inputs, inputCache) {
	var cacheEntry = inputCache.cache[callsite];
	assert(cacheEntry !== undefined, 'Attempting to normalize NN inputs for which no sample stats exist!');
	// if (callsite === 'stateFeatures') {
	// 	console.log('stateFeatures means and stddevs:');
	// 	console.log(cacheEntry.means);
	// 	console.log(cacheEntry.stddevs);
	// 	console.log('inputs received:');
	// 	console.log(inputs);
	// }
	// return numeric.diveq(numeric.sub(inputs, cacheEntry.mins), cacheEntry.ranges);
	return numeric.diveq(numeric.sub(inputs, cacheEntry.means), cacheEntry.stddevs);
}


module.exports = {
	makeInputSampleCache: makeInputSampleCache,
	collectSample: collectSample,
	normalizeInputs: normalizeInputs
};

