
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
	var xbase = 0;
	var xform = new THREE.Matrix4();
	var totalgeo = new Geo.Geometry();
	for (var i = 0; i < geometries.length; i++) {
		var geo = geometries[i];
		var size = geo.getbbox().size();
		xform.makeTranslation(xbase + size.x/2, 0, 0);
		xbase += size.x + padding;
		totalgeo.mergeWithTransform(geo, xform);
	}
	saveOBJ(totalgeo, filename);
}

module.exports = {
	saveOBJ: saveOBJ,
	saveLineup: saveLineup
};