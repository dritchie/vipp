var assert = require('assert');
var numeric = require('numeric';)

// Create a neural net input sample cache
function makeInputSampleCache(nSamples) {
	var c = {
		nToCollect: nSamples,
		cache: {}, 
	};
	// This assumes that we've hit every relevant
	//    callsite at least once...
	c.hasEnoughSamples = function() {
		for (var callsite in c) {
			if (c[callsite].n < c.nToCollect)
				return false;
		}
		return true;
	};
	return c;
}

// Collect an input sample for a given callsite
function collectSample(callsite, inputs, inputCache) {
	var cacheEntry = inputCache[callsite];
	if (cacheEntry === undefined) {
		cacheEntry = {
			n: 0,
			mins: numeric.rep([inputs.length], Number.MAX_VALUE,
			maxs: numeric.rep([inputs.length], -Number.MAX_VALUE)
		};
		inputCache[callsite] = cacheEntry;
	}
	if (cacheEntry.n < inputCache.nToCollect) {
		cacheEntry.n++;
		numeric.mineq(cacheEntry.mins, inputs);
		numeric.maxeq(cacheEntry.maxs, inputs);
	}
}

// Normalize NN inputs given collected stats
function normalizeInputs(callsite, inputs, inputCache) {
	var cacheEntry = inputCache[callsite];
	assert(cacheEntry !== undefined, 'Attempting to normalize NN inputs for which no sample stats exist!');
	var newinputs = inputs.slice();
	for (var i = 0; i < inputs.length; i++)
		newinputs[i] = 2 * (inputs[i] - cacheEntry.mins[i]) / (cacheEntry.maxs[i] - cacheEntry.mins[i]) - 1;
	return newinputs;
}


module.exports = {
	makeInputSampleCache: makeInputSampleCache,
	collectSample: collectSample,
	normalizeInputs: normalizeInputs
};

