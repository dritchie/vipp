
var utils = require('../lib/utils');
var Geo = require('../lib/geometry');

function genAndSave(guide, params, n, basefilename) {
	for (var i = 0; i < n; i++) {
		var geolist = guide('', params);
		var accumgeo = new Geo.Geometry();
		for (var j = 0; j < geolist.length; j++)
			accumgeo.merge(geolist[j]);
		var size = accumgeo.getbbox().size();
		console.log(size.x, size.z);
		utils.saveOBJ(accumgeo, basefilename + i + '.obj');
	}
}

module.exports = {
	genAndSave: genAndSave
};