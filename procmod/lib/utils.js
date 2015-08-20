
var fs = require('fs');
var THREE = require('three');
var OBJLoader = require('./OBJLoader');
var OBJExporter = require('./OBJExporter');
var Geo = require('./geometry');
var Grids = require('./grids')

function loadOBJ(filename) {
	var filetext = fs.readFileSync(filename).toString();
	var parser = new OBJLoader();
	return parser.parse(filetext);
}

function loadVolumeTarget(filename) {
	var voxparams = {
		percentSameSigma: 0.005,
		percentOutsideSigma: 0.005,
		size: 0.25,
		bounds: null,
		targetGrid: new Grids.BinaryGrid3()
	};

	var obj = loadOBJ(filename);
	var bufgeo = obj.children[0].geometry;
	var geom = new THREE.Geometry();
	geom.fromBufferGeometry(bufgeo);
	var mygeom = new Geo.Geometry();
	mygeom.fromThreeGeo(geom);
	voxparams.bounds = mygeom.getbbox().clone();
	voxparams.bounds.expandByScalar(0.1);
	voxparams.targetGrid.clearall();
	mygeom.voxelize(voxparams.targetGrid, voxparams.bounds, voxparams.size, true);

	return voxparams;
};

function saveOBJ(geo, filename) {
	var threegeo = geo.toThreeGeo();
	var mesh = new THREE.Mesh(threegeo, null);
	var exporter = new OBJExporter();
	var str = exporter.parse(mesh);
	fs.writeFileSync(filename, str);
}

module.exports = {
	loadOBJ: loadOBJ,
	loadVolumeTarget: loadVolumeTarget,
	saveOBJ: saveOBJ
};