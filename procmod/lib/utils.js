
var fs = require('fs');
var THREE = require('three');
var OBJExporter = require('./OBJExporter');
var Geo = require('./geometry');

function saveOBJ(geo, filename) {
	var threegeo = geo.toThreeGeo();
	var mesh = new THREE.Mesh(threegeo, null);
	var exporter = new OBJExporter();
	var str = exporter.parse(mesh);
	fs.writeFileSync(filename, str);
}

function saveLineup(geometries, filename) {
	var padding = 1;
	var xcenter = 0;
	var xform = new THREE.Matrix4();
	var totalgeo = new Geo.Geometry();
	for (var i = 0; i < geometries.length; i++) {
		var geo = geometries[i];
		var size = geo.getbbox().size();
		var center = geo.getbbox().center();
		xcenter += size.x/2;
		xform.makeTranslation(xcenter-center.x, 0, 0);
		totalgeo.mergeWithTransform(geo, xform);
		xcenter += size.x/2 + padding;
	}
	var center = totalgeo.getbbox().center();
	xform.makeTranslation(-center.x, 0, 0);
	totalgeo.transform(xform);
	saveOBJ(totalgeo, filename);
}

module.exports = {
	saveOBJ: saveOBJ,
	saveLineup: saveLineup
};