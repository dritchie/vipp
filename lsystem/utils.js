var fs = require('fs');
var Canvas = require('canvas');
var THREE = require('three');
var assert = require('assert');

function render(canvas, viewport, branches) {
	if (viewport === undefined)
		viewport = {xmin: 0, ymin: 0, xmax: canvas.width, ymax: canvas.height};

	function world2img(p) {
		return new THREE.Vector2(
			canvas.width * (p.x - viewport.xmin) / (viewport.xmax - viewport.xmin),
			canvas.height * (p.y - viewport.ymin) / (viewport.ymax - viewport.ymin)
		);
	}

	var ctx = canvas.getContext('2d');

	// Fill background
	ctx.rect(0, 0, canvas.width, canvas.height);
	ctx.fillStyle = 'white';
	ctx.fill();

	// Draw
	ctx.strokeStyle = 'black';
	ctx.lineCap = 'round';
	for (var i = 0; i < branches.length; i++) {
		var branch = branches[i];
		var istart = world2img(branch.start);
		var iend = world2img(branch.end);
		var iwidth = branch.width / (viewport.xmax - viewport.xmin) * canvas.width;
		ctx.beginPath();
		ctx.lineWidth = iwidth;
		ctx.moveTo(istart.x, istart.y);
		ctx.lineTo(iend.x, iend.y);
		ctx.stroke();
	}
}

function renderOut(filename, res, viewport, branches) {
	var canvas = new Canvas(res.width, res.height);
	render(canvas, viewport, branches);
	fs.writeFileSync(filename, canvas.toBuffer());
}


function ImageData2D(canvas) {
	this.data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
	this.width = canvas.width;
	this.height = canvas.height;
}
ImageData2D.prototype = {
	constructor: ImageData2D,
	getPixel: function(x, y) {
		var i = y*this.width + x;
		return [this.data[4*i], this.data[4*i+1], this.data[4*i+2], this.data[4*i+3]];
	},
	getLuminance: function(x, y) {
		var i = y*this.width + x;
		return 0.2126*this.data[4*i] + 0.7152*this.data[4*i+1] + 0.0722*this.data[4*i+2];
	},
	isFilled: function(x, y) {
		var i = y*this.width + x;
		return this.data[4*i] !== 255 || this.data[4*i+1] !== 255 || this.data[4*i+2] !== 255; 
	},
	percentFilled: function() {
		var filled = 0;
		for (var y = 0; y < this.height; y++) {
			for (var x = 0; x < this.width; x++) {
				filled += this.isFilled(x, y);
			}
		}
		return filled / (this.width * this.height);
	},
	bilateralSymmetryHoriz: function() {
		// Horizonal bilateral symmetry
		var sim = 0;
		for (var y = 0; y < this.height; y++) {
			for (var x = 0; x < this.width / 2; x++) {
				var f1 = this.isFilled(x, y);
				var f2 = this.isFilled(this.width - x - 1, y);
				sim += (f1 === f2);
			}
		}
		return sim / (0.5 * this.width * this.height);
	},
	filledBilateralSymmetryHoriz: function() {
		// Horizonal bilateral symmetry
		var sim = 0;
		var n = 0;
		for (var y = 0; y < this.height; y++) {
			for (var x = 0; x < this.width / 2; x++) {
				var f1 = this.isFilled(x, y);
				var f2 = this.isFilled(this.width - x - 1, y);
				if (f1 || f2) {
					sim += (f1 === f2);
					n++;
				}
			}
		}
		return sim / n;
	}
};


var TempSchedules = {

	linear: function(i, n) {
		return Math.max(i/n, 0.001);
	},
	linearStop: function(stop) {
		return function(i, n) {
			var stopi = stop*n;
			return Math.min(1, Math.max(i/stopi, 0.001));
		}
	},
	asymptotic: function(rate) {
		return function(i, n) {
			var x = i/n + 0.001;
			return Math.max(0.001, 1 + (1 / (-rate*x)));
		}
	}

};


module.exports = {
	render: render,
	renderOut: renderOut,
	newImageData2D: function(canvas) { return new ImageData2D(canvas); },
	newCanvas: function(w, h) { return new Canvas(w, h); },
	TempSchedules: TempSchedules,
	require: require	// so that webppl code can just require whatever code it wants
};






