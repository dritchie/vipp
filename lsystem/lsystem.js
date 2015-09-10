var utils = require('lsystem/utils');
var THREE = require('three');


var polar2rect = function(r, theta) {
	return new THREE.Vector2(r*Math.cos(theta), r*Math.sin(theta));
}

var line = function(start, angle, width, length) {
	return {
		start: start,
		angle: angle,
		width: width,
		end: start.clone().add(polar2rect(length, angle))
	};
};

var branch = function(depth, currBranch, branches) {
	var width = 0.9 * currBranch.width;
	var length = 2;
	var newang = gaussian(currBranch.angle, Math.PI/8);
	var newbranch = line(currBranch.end, newang, width, length);
	branches.push(newbranch);
	// Terminate?
	if (!flip(Math.exp(-0.045*depth)))
		return;
	// Continue or fork?
	if (flip(0.5)) {
		branch(depth + 1, newbranch, branches);
	} else {
		var seed1 = line(newbranch.end, newbranch.angle - Math.abs(gaussian(0, Math.PI/6)), newbranch.width, 0);
		branch(depth + 1, seed1, branches);
		var seed2 = line(newbranch.end, newbranch.angle + Math.abs(gaussian(0, Math.PI/6)), newbranch.width, 0);
		branch(depth + 1, seed2, branches);
	}
};

var generate = function() {
	var branches = [];
	branch(0, line(new THREE.Vector2(0, 0), -Math.PI/2, 0.75, 0), branches);
	return branches;
};

// TEST
var branches = generate();
console.log(branches.length);
var bbox = new THREE.Box2();
for (var i = 0; i < branches.length; i++) {
	var br = branches[i];
	bbox.expandByPoint(br.start);
	bbox.expandByPoint(br.end);
}
console.log(bbox);
utils.renderOut(
	'lsystem/results/lsystem.png',
	{width: 600, height: 600},
	{xmin: -20, xmax: 20, ymin: -40, ymax: 0},
	// {xmin: 0, xmax: 40, ymin: 0, ymax: 40},
	branches
);