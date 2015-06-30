
// Must be run from the repo root.

var runExperiment = require('./harness').runExperiment;

function makeConditions(nameGuides, options) {
	return nameGuides.map(function(ng) {
		return {
			name: ng.name,
			guide: ng.guide,
			opts: options
		};
	})
}

// ----------------------------------------------------------------------------

// // Gamma-gaussian

// var experiment_options = {
// 	outBaseName: './experiments/convergence/test',
// 	numRuns: 20
// };

// var condition_options = {
// 	nSamples: 100,
// 	nSteps: 5000,
// 	convergeEps: 0.1,
// 	// initLearnRate: 0.5
// 	initLearnRate: 0.25
// };

// var conditions = makeConditions([
// 	{name: 'Mean-field', guide: 'guide_meanField'},
// 	// {name: 'Mean-field + Bounds', guide: 'guide_bounds'},
// 	{name: 'Mean-field + Backprop', guide: 'guide_backprop'}
// ], condition_options);

// runExperiment('./experiments/convergence/gammaGaussian.js', conditions, experiment_options);

// ----------------------------------------------------------------------------

// Gaussian sum

var experiment_options = {
	outBaseName: './experiments/convergence/test',
	numRuns: 20
};

var condition_options = {
	nSamples: 100,
	nSteps: 5000,
	convergeEps: 0.1,
	// initLearnRate: 0.5
	initLearnRate: 0.25
};

var conditions = makeConditions([
	{name: 'Mean-field', guide: 'guide_meanField'},
	{name: 'Context-sensitive', guide: 'guide_context'}
], condition_options);

runExperiment('./experiments/convergence/gaussianSum.js', conditions, experiment_options);

