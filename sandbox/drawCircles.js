
var dot = function(p0, p1) {
	return p0[0]*p1[0] + p0[1]*p1[1];
}

var norm = function(p) {
	return Math.sqrt(dot(p, p));
}

// Rendering outputs
var drawCircles = function(points, radius, imgw, imgh, view, outfilename) {
	var Canvas = require('canvas')
	  , canvas = new Canvas(imgw, imgh)
	  , ctx = canvas.getContext('2d')
	  , fs = require('fs');

	// Fill background
	ctx.rect(0, 0, imgw, imgh);
	ctx.fillStyle = 'white';
	ctx.fill();

	// Determine 2D viewing transform
	var minx, maxx, miny, maxy, xlen, ylen;
	if (view !== undefined) {
		minx = view[0];
		miny = view[1];
		maxx = view[2];
		maxy = view[3];
		xlen = maxx - minx;
		ylen = maxy - miny;
	} else {
		// Find the bbox of all the circles
		minx = Infinity, maxx = 0, miny = Infinity, maxy = 0;
		for (var i = 0; i < points.length; i++) {
			var p = points[i];
			minx = Math.min(minx, p[0] - radius);
			maxx = Math.max(maxx, p[0] + radius);
			miny = Math.min(miny, p[1] - radius);
			maxy = Math.max(maxy, p[1] + radius);
		}
		// Pad a little bit
		var padAmt = Math.max(maxx - minx, maxy - miny) * 0.05;
		minx -= padAmt; maxx += padAmt; miny -= padAmt; maxy += padAmt;
		xlen = maxx - minx;
		ylen = maxy - miny;
		// Correct aspect ratio
		var imga = imgw / imgh;
		var worlda = xlen / ylen;
		if (imga <= 1.0) {
			if (worlda < imga) {
				// Set width to achieve correct aspect
				xlen = ylen * imga;
			} else {
				// Set height to achieve correct aspect
				ylen = xlen / imga;
			}
		} else if (imga > 1.0) {
			if (worlda > imga) {
				// Set height to achieve correct aspect
				ylen = xlen / imga;
			} else {
				// Set width to achieve correct aspect
				xlen = ylen * imga;
			}
		}
		// Finalize
		var origxlen = maxx - minx;
		var extrax = (xlen - origxlen) / 2;
		minx -= extrax; maxx += extrax;
		var origylen = maxy - miny;
		var extray = (ylen- origylen) / 2;
		miny -= extray; maxy += extray;
	}

	// Transformation from world space to image space
	var world2img = function(p) {
		var pxnorm = (p[0] - minx) / xlen;
		var pynorm = (p[1] - miny) / ylen;
		return [
			pxnorm * imgw,
			pynorm * imgh
		];
	}

	// Image space radius
	var imgr = norm(world2img([minx+radius, miny]));

	// Draw circles for each point
	for (var i = 0; i < points.length; i++) {
		var p = world2img(points[i]);
		ctx.beginPath();
		ctx.arc(p[0], p[1], imgr, 0, 2*Math.PI, true);
		ctx.fillStyle = 'blue';
		ctx.fill();
		ctx.strokeStyle = 'black';
		ctx.lineWidth = 4;
		ctx.stroke();
	}

	// Save to file
	fs.writeFileSync(outfilename, canvas.toBuffer());
}

// // TEST
// drawCircles([
// 	[-1, -1],
// 	[1, -1],
// 	[1, 1],
// 	[-1, 1]
// ],
// 0.1, 800, 800, 'circles.png');


module.exports = {
	drawCircles: drawCircles
};


