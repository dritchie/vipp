var drawCircles = require('sandbox/drawCircles.js').drawCircles;

var nPoints = 6;
var targetDist = 0.5;
var tightness = 0.01;

var add = function(p0, p1) {
	return [
		p0[0] + p1[0],
		p0[1] + p1[1]
	];
}

var sub = function(p0, p1) {
	return [
		p0[0] - p1[0],
		p0[1] - p1[1]
	];
}

var dot = function(p0, p1) {
	return p0[0]*p1[0] + p0[1]*p1[1];
}

var norm = function(p) {
	return Math.sqrt(dot(p, p));
}

var dist = function(p0, p1) {
	return norm(sub(p0, p1));
}

var rotate = function(v, ang) {
	var cosang = Math.cos(ang);
	var sinang = Math.sin(ang);
	return [
		cosang * v[0] - sinang * v[1],
		sinang * v[0] + cosang * v[1]
	];
}

var target = function() {
	// Sample some random points
	var points = [];
	for (var i = 0; i < nPoints; i++) {
		points.push([
			gaussian(2*i, 0.0, 2.0),
			gaussian(2*i+1, 0.0, 2.0)
		]);
	}
	// Encourage subsequent points to be some distance apart
	for (var i = 0; i < nPoints; i++) {
		var p0 = points[i];
		var p1 = points[(i+1) % nPoints];
		var d = dist(p0, p1);
		factor(gaussianERP.score([targetDist, tightness], d));
	}
	// Encourage point triples to be as-linear-as-possible
	for (var i = 0; i < nPoints; i++) {
		var p0 = points[i];
		var p1 = points[(i+1) % nPoints];
		var p2 = points[(i+2) % nPoints];
		var v0 = sub(p0, p1);
		var v1 = sub(p2, p1);
		var d = dot(v0, v1) / (norm(v0) * norm(v1));
		factor(gaussianERP.score([-1.0, tightness], d));
	}
	return points;
}

function pnnSample(params) {
	return Math.log(gammaERP.sample(params));
}

// True radius of the circle is given by:
var theta = 2*Math.PI/nPoints;
var truerad = (targetDist / Math.sin(theta)) * Math.sin(0.5*(Math.PI - theta));
// Hand-designed guide program
var guide_handTuned = function(params) {
	var p = function() { return param(params, undefined, undefined, gaussianERP.sample, [0.0, 1.0]); };
	var pnn = function() { return param(params, undefined, Math.exp, gaussianERP.sample, [0.0, 1.0]); };
	var points = [];
	// First point is purely random
	points.push([
		// gaussian(p(), Math.abs(p())),
		// gaussian(p(), Math.abs(p()))
		gaussian(p(), pnn()),
		gaussian(p(), pnn())
	]);
	// Pick the center of the circle to be directly below this point
	var rvec = [0.0, truerad];
	var c = sub(points[0], rvec);
	// Pick the next n-1 points as random perturbations around 
	//    exact radial offsets
	for (var i = 1; i < nPoints; i++) {
		var ang = i * theta;
		var offset = rotate(rvec, ang);
		var ploc = add(c, offset);
		points.push([
			// gaussian(ploc[0], Math.abs(p())),
			// gaussian(ploc[1], Math.abs(p()))
			gaussian(ploc[0], pnn()),
			gaussian(ploc[1], pnn())
		]);
	}
	return points;
}

// Mean-field guide program
var guide_meanField = function(params) {
	var p = function() { return param(params, undefined, undefined, gaussianERP.sample, [0.0, 1.0]); };
	var pnn = function() { return param(params, undefined, Math.exp, gaussianERP.sample, [0.0, 1.0]); };
	// var pnn = function() { return param(params, undefined, Math.exp, pnnSample, [1.0, 1.0]); };
	// Sample some random points
	var points = [];
	for (var i = 0; i < nPoints; i++) {
		// var sp1 = p();
		// var sp2 = p();
		points.push([
			// gaussian(p(), Math.abs(p())),
			// gaussian(p(), Math.abs(p()))
			gaussian(p(), pnn()),
			gaussian(p(), pnn())
		]);
	}
	return points;
}

var flattenPoints = function(points) {
	var lst = [];
	for (var i = 0; i < points.length; i++) {
		var p = points[i];
		lst.push(p[0]);
		lst.push(p[1]);
	}
	return lst;
}

// Perceptron with single hidden layer
var sigmoid = function(x) {
	return 1.0 / (1.0 + Math.exp(-x));
}
var perceptron = function(param, inputs, nHidden) {
	var nInputs = inputs.length;
	// Compute activation at each hidden node
	var hiddens = [];
	for (var h = 0; h < nHidden; h++) {
		var sum = param(); // bias
		for (var i = 0; i < nInputs; i++) {
			sum = sum + param() * inputs[i];
		}
		hiddens.push(sigmoid(sum));
	}
	// Compute output
	var output = param(); // bias
	for (var h = 0; h < nHidden; h++) {
		output = output + param() * hiddens[h];
	}
	return output;
}

// Radial basis function network
var rbfn = function(param, inputs, nHidden) {
	var nInputs = inputs.length;
	// Compute activation at each hidden node
	var hiddens = [];
	for (var h = 0; h < nHidden; h++) {
		var sum = 0.0;
		for (var i = 0; i < nInputs; i++) {
			var diff = inputs[i] - param();
			sum = sum + diff * diff;
		}
		// sum = sum * Math.abs(param());
		sum = sum * Math.exp(param());
		hiddens.push(Math.exp(-sum));
	}
	// Compute output
	var output = param();
	for (var h = 0; h < nHidden; h++) {
		output = output + param() * hiddens[h];
	}
	return output;
}

// Linear 'network'
var linearnet = function(param, inputs) {
	var sum = param();
	for (var i = 0; i < inputs.length; i++)
		sum = sum + param() * inputs[i];
	return sum;
}

// Guide program using neural net to compute ERP params
var makeGuideNN = function(nn, nHidden) {
	return function (params) {
		var p = function() { return param(params, undefined, undefined, gaussianERP.sample, [0.0, 1.0]); };
		// var pnn = function() { return param(params, undefined, Math.exp, gaussianERP.sample, [0.0, 1.0]); };
		var pnn = function() { return param(params, undefined, Math.abs, gaussianERP.sample, [0.0, 1.0]); };
		var points = [];
		points.push([
			gaussian(p(), pnn()),
			gaussian(p(), pnn())
		]);
		for (var i = 0; i < nPoints-1; i++) {
			// var inputs = flattenPoints(points);
			var inputs = points[0];   // Just the first point
			var muX = nn(p, inputs, nHidden);
			var sigmaX =  Math.abs(nn(p, inputs, nHidden));
			// var sigmaX =  Math.exp(nn(p, inputs, nHidden));
			var muY = nn(p, inputs, nHidden);
			var sigmaY = Math.abs(nn(p, inputs, nHidden));
			// var sigmaY = Math.exp(nn(p, inputs, nHidden));
			points.push([
				gaussian(muX, sigmaX),
				gaussian(muY, sigmaY)
			]);
		}
		return points;
	};
}

// Different neural net guides
var nHidden = 1;
// var nHidden = 10;
var guide_perceptron = makeGuideNN(perceptron, nHidden);
var guide_rbfn = makeGuideNN(rbfn, nHidden);
var guide_linear = makeGuideNN(linearnet);

// Guide that expresses the minimal relationship that I think is needed
//    to capture the right behavior.
var guide_minimal = function(params) {
	var pcount = 0;
	var p = function() { pcount++; return param(pcount, params, undefined, undefined, gaussianERP.sample, [0.0, 1.0]); };
	var pnn = function() { pcount++; return param(pcount, params, undefined, Math.exp, gaussianERP.sample, [0.0, 1.0]); };
	var points = [];
	points.push([
		gaussian(0, p(), pnn()),
		gaussian(1, p(), pnn())
	]);
	for (var i = 1; i < nPoints; i++) {
		// var inputs = points[0];
		var inputs = points[i-1];
		// var muX = inputs[0] + p();
		var muX = p()*inputs[0] + p();
		// var muX = p()*inputs[0] + p()*inputs[1] + p();
		var sigmaX = pnn();
		// var muY = inputs[1] + p();
		var muY = p()*inputs[1] + p();
		// var muY = p()*inputs[1] + p()*inputs[0] + p();
		var sigmaY = pnn();
		points.push([
			gaussian(2*i, muX, sigmaX),
			gaussian(2*i+1, muY, sigmaY)
		]);
	}
	return points;
};

// ----------------------------------------------------------------------------

// Which guide we're actually going to use
// var guide = guide_handTuned;
// var guide = guide_meanField;
// var guide = guide_perceptron;
// var guide = guide_rbfn;
// var guide = guide_linear;
var guide = guide_minimal;

console.time('time');
var result = infer(target, guide, undefined, {
// var result = require('experiments/convergence/harness.js').run('sandbox/convergence.csv', target, guide, undefined, {
	verbosity: 2,
	// nSamples: 10,
	nSamples: 100,
	nSteps: 1000,
	// nSteps: 20,
	// convergeEps: 0.1,
	convergeEps: 0.05,
	// initLearnRate: 1.0
	initLearnRate: 0.5,
	// initLearnRate: 0.25,
	// initLearnRate: 0.1,
	// regularize: {
	// 	method: 'L1',
	// 	weight: 10.0
	// },
	gradientOpts: {
		// method: 'ELBO',
		// method: 'EUBO',
		method: 'ELBO|EUBO',
		// nChains: 8,
		nChains: 32,
		burnIn: 1000,
		lag: 0,
		mixWeight: 0.5
	}
});
console.timeEnd('time');
// if (result.euboChainStates !== undefined) {
// 	for (var i = 0; i < result.euboChainStates.length; i++) {
// 		drawCircles(result.euboChainStates[i], 0.1, 800, 800, [-3, -3, 3, 3], 'chainState_'+i+'.png');
// 	}
// }
for (var i = 0; i < 10; i++) {
	var points = guide(result.params);
	drawCircles(points, 0.1, 800, 800, [-3, -3, 3, 3], 'sandbox/sample_' + i + '.png');
}
return result;



