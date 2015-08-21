
var utils = require('../lib/utils');
var Geo = require('../lib/geometry');
var THREE = require('three');

function genAndSave(guide, params, n, filename) {
	var padding = 1;
	var xbase = 0;
	var xform = new THREE.Matrix4();
	var totalgeo = new Geo.Geometry();
	for (var i = 0; i < n; i++) {
		var geolist = guide('', params);
		var accumgeo = new Geo.Geometry();
		for (var j = 0; j < geolist.length; j++)
			accumgeo.merge(geolist[j]);
		var size = accumgeo.getbbox().size();
		console.log(size.x, size.z);
		xform.makeTranslation(xbase + size.x/2, 0, 0);
		xbase += size.x + padding;
		accumgeo.transform(xform);
		totalgeo.merge(accumgeo);
	}
	utils.saveOBJ(totalgeo, filename);
}

module.exports = {
	genAndSave: genAndSave
};