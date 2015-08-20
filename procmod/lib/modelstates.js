
var THREE = require('three');
var Geo = require('./geometry.js');
var Grids = require('./grids.js');

var ModelStates = {
	Sequential: {},
	Compositional: {}
};

// ------------------------------------------------------------------------

ModelStates.Sequential.Voxelizing = function(voxparams)
{
	this.geometry = null;
	this.grid = null;
	this.bbox = null;
	this.next = null;
	this.length = 0;
	this.score = 0;
	this.voxparams = voxparams;
}

// Voxparams has:
//   - 'bounds'
//   - one of 'size' or 'dims'
ModelStates.Sequential.Voxelizing.create = function(voxparams)
{
	var ms = new ModelStates.Sequential.Voxelizing(voxparams);
	ms.geometry = new Geo.Geometry();
	if (voxparams.dims === undefined)
		voxparams.dims = voxparams.bounds.size().divideScalar(voxparams.size).ceil();
	ms.grid = new Grids.BinaryGrid3(voxparams.dims);
	ms.bbox = new THREE.Box3();
	return ms;
}

ModelStates.Sequential.Voxelizing.extend = function(geo, next, doupdate)
{
	var ms = new ModelStates.Sequential.Voxelizing(next.voxparams);
	ms.geometry = geo;
	if (doupdate)
	{
		ms.grid = next.grid.clone();
		geo.voxelize(ms.grid, ms.voxparams.bounds, ms.grid.dims, true);
		ms.bbox = geo.getbbox().clone().union(next.bbox);
	}
	ms.next = next;
	ms.length = 1 + next.length;
	return ms;
}

ModelStates.Sequential.Voxelizing.prototype = 
{
	constructor: ModelStates.Sequential.Voxelizing,

	addGeometry: function(geo)
	{
		var newstate;
		// We don't even bother updating the state if the score is already -Infinity.
		// (This is sort of like bailing out early: the program runs to completion, but
		//   it doesn't do any of the really expensive stuff).
		// If adding this new geometry results in a self-intersection, then
		//    the score immediately drops to log(0).
		if (this.score === -Infinity || this.intersects(geo))
		{
			newstate = ModelStates.Sequential.Voxelizing.extend(geo, this, false);
			newstate.score = -Infinity;
		}
		else
		{
			newstate = ModelStates.Sequential.Voxelizing.extend(geo, this, true);
			var vp = this.voxparams;
			var percentSame = vp.targetGrid.percentCellsEqualPadded(newstate.grid);
			var targetExtent = vp.bounds.size();
			var extralo = vp.bounds.min.clone().sub(newstate.bbox.min).clampScalar(0, Infinity).divide(targetExtent);
			var extrahi = newstate.bbox.max.clone().sub(vp.bounds.max).clampScalar(0, Infinity).divide(targetExtent);
			var percentOutside = extralo.x + extralo.y + extralo.z + extrahi.x + extrahi.y + extrahi.z;
			newstate.score = gaussianERP.score([1, vp.percentSameSigma], percentSame) +
							 gaussianERP.score([0, vp.percentOutsideSigma], percentOutside);
		}
		return newstate;
	},

	// The linear chain of states acts like a one-dimesional BVH.
	intersects: function(geo)
	{
		var geobbox = geo.getbbox();
		// Walk down the chain of states, looking for intersections.
		for (var currstate = this; currstate !== null; currstate = currstate.next)
		{
			// If at any point the new geo's bbox no longer intersects the accumulated state
			//    bbox, we can bail out with a false.
			if (!geobbox.isIntersectionBox(currstate.bbox))
				return false;
			// If we find an intersection, bail out with a true.
			if (geo.intersects(currstate.geometry))
				return true;
		}
		return false;
	},

	getCompleteGeometry: function()
	{
		var accumgeo = new Geo.Geometry();
		for (var currstate = this; currstate !== null; currstate = currstate.next)
			accumgeo.merge(currstate.geometry);
		return accumgeo;
	}
}

// ------------------------------------------------------------------------

// Simplified sequential state that doesn't do volume matching
ModelStates.Sequential.NonIntersecting = function()
{
	this.geometry = null;
	this.bbox = null;
	this.next = null;
	this.length = 0;
	this.score = 0;
}

ModelStates.Sequential.NonIntersecting.create = function()
{
	var ms = new ModelStates.Sequential.NonIntersecting();
	ms.geometry = new Geo.Geometry();
	ms.bbox = new THREE.Box3();
	return ms;
}

ModelStates.Sequential.NonIntersecting.extend = function(geo, next, doupdate)
{
	var ms = new ModelStates.Sequential.NonIntersecting();
	ms.geometry = geo;
	if (doupdate)
		ms.bbox = geo.getbbox().clone().union(next.bbox);
	ms.next = next;
	ms.length = 1 + next.length;
	return ms;
}

ModelStates.Sequential.NonIntersecting.prototype = 
{
	constructor: ModelStates.Sequential.NonIntersecting,

	addGeometry: function(geo)
	{
		var newstate;
		// We don't even bother updating the state if the score is already -Infinity.
		// (This is sort of like bailing out early: the program runs to completion, but
		//   it doesn't do any of the really expensive stuff).
		// If adding this new geometry results in a self-intersection, then
		//    the score immediately drops to log(0).
		if (this.score === -Infinity || this.intersects(geo))
		{
			newstate = ModelStates.Sequential.NonIntersecting.extend(geo, this, false);
			newstate.score = -Infinity;
		}
		else
		{
			newstate = ModelStates.Sequential.NonIntersecting.extend(geo, this, true);
			newstate.score = 0;
		}
		return newstate;
	},

	// The linear chain of states acts like a one-dimesional BVH.
	intersects: function(geo)
	{
		var geobbox = geo.getbbox();
		// Walk down the chain of states, looking for intersections.
		for (var currstate = this; currstate !== null; currstate = currstate.next)
		{
			// If at any point the new geo's bbox no longer intersects the accumulated state
			//    bbox, we can bail out with a false.
			if (!geobbox.isIntersectionBox(currstate.bbox))
				return false;
			// If we find an intersection, bail out with a true.
			if (geo.intersects(currstate.geometry))
				return true;
		}
		return false;
	},

	getCompleteGeometry: function()
	{
		var accumgeo = new Geo.Geometry();
		for (var currstate = this; currstate !== null; currstate = currstate.next)
			accumgeo.merge(currstate.geometry);
		return accumgeo;
	}
}

// ------------------------------------------------------------------------

module.exports = ModelStates;



