
function drawPalette(palette, filename) {
	var width = 100;
	var height = 200;

	var Canvas = require('canvas')
	  , canvas = new Canvas(width*palette.length, height)
	  , ctx = canvas.getContext('2d')
	  , fs = require('fs');

	for (var i = 0; i < palette.length; i++) {
		var color = palette[i];
		ctx.fillStyle = 'rgb(' +
			Math.floor(color[0]*255) + ',' + 
			Math.floor(color[1]*255) + ',' + 
			Math.floor(color[2]*255) + ')';
		ctx.fillRect(i*width, 0, width, height);
	}

	fs.writeFileSync(filename, canvas.toBuffer());
}

module.exports = {
	drawPalette: drawPalette
};