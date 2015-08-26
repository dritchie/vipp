var ad = require('../ad/functions');
var assert = require('assert');

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
			mins: [],
			maxs: []
		};
		for (var i = 0; i < inputs.length; i++) {
			cacheEntry.mins.push(Number.MAX_VALUE);
			cacheEntry.maxs.push(-Number.MAX_VALUE);
		}
		inputCache[callsite] = cacheEntry;
	}
	if (cacheEntry.n < inputCache.nToCollect) {
		cacheEntry.n++;
		for (var i = 0; i < inputs.length; i++) {
			cacheEntry.mins[i] = Math.min(cacheEntry.mins[i], ad.ad_primal(inputs[i]));
			cacheEntry.maxs[i] = Math.max(cacheEntry.maxs[i], ad.ad_primal(inputs[i]));
		}
	}
}

// Normalize NN inputs given collected stats
function normalizeInputs(callsite, inputs, inputCache) {
	var cacheEntry = inputCache[callsite];
	assert(cacheEntry !== undefined, 'Attempting to normalize NN inputs for which no sample stats exist!');
	for (var i = 0; i < inputs.length; i++) {
		inputs[i] = 2 * (inputs[i] - cacheEntry.mins[i]) / (cacheEntry.maxs[i] - cacheEntry.mins[i]) - 1;
	}
}


module.exports = {
	makeInputSampleCache: makeInputSampleCache,
	collectSample: collectSample,
	normalizeInputs: normalizeInputs
};

