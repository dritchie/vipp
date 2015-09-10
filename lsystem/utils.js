var fs = require('fs');
var Canvas = require('canvas');
var THREE = require('three');

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
	branches.forEach(function(branch) {
		var istart = world2img(branch.start);
		var iend = world2img(branch.end);
		var iwidth = branch.width / (viewport.xmax - viewport.xmin) * canvas.width;
		ctx.beginPath();
		ctx.lineWidth = iwidth;
		ctx.moveTo(istart.x, istart.y);
		ctx.lineTo(iend.x, iend.y);
		ctx.stroke();
	});
}

function renderOut(filename, res, viewport, branches) {
	var canvas = new Canvas(res.width, res.height);
	render(canvas, viewport, branches);
	fs.writeFileSync(filename, canvas.toBuffer());
}



module.exports = {
	render: render,
	renderOut: renderOut
};