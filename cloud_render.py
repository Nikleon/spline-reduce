from math import pi, sin, cos


from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3, LColorf

from panda3d.core import GeomVertexData, GeomVertexFormat, GeomVertexWriter, Thread, GeomNode, Geom, RenderState, GeomPrimitive, GeomPoints, NodePath, GeomEnums

import numpy as np
from math import sin, cos, pi
import random

#TEMP inclusion
def generate_torus_cloud(
		n			= 10_000,
		center		= np.array([0, 0, 0]),
		inner_radius= 8, 
		outer_radius= 15,
		noise_st_dev= 0.1):
	points = []
	for i in range(n):
		outer_deg = random.uniform(0, 2 * pi)
		inner_deg = random.uniform(0, 2 * pi)
		# TODO: bias towards one POV

		radial = outer_radius + inner_radius * cos(inner_deg)
		z = inner_radius * sin(inner_deg)

		points.append(
			tuple((
				np.array([
					radial * cos(outer_deg),
					
					z,
					radial * sin(outer_deg)
				]) + np.random.normal(loc=center, scale=noise_st_dev , size=(3, ))
			).tolist())
		)
	return points

def make_cloud_node(pts):
	ptCloudData = GeomVertexData("point cloud data", GeomVertexFormat.getV3c4(), GeomEnums.UH_static)
	vertexWriter = GeomVertexWriter(ptCloudData, Thread.getCurrentThread())
	vertexWriter.setColumn("vertex")
	colorWriter = GeomVertexWriter(ptCloudData, Thread.getCurrentThread())
	colorWriter.setColumn("color")

	for (x, y, z) in pts:
		vertexWriter.addData3(x, y, z)
		colorWriter.addData4(1.0, 0.0, 0.0, 1.0)

	geomPts = GeomPoints(GeomEnums.UH_static)
	geomPts.addConsecutiveVertices(0, len(pts))
	geomPts.closePrimitive()

	geom = Geom(ptCloudData)
	geom.addPrimitive(geomPts)

	node = GeomNode("point cloud")
	node.addGeom(geom, RenderState.makeEmpty())
	return node


class MyApp(ShowBase):

	def __init__(self):
		ShowBase.__init__(self)

		self.win.setClearColorActive(True)
		self.win.setClearColor(LColorf(0.0, 0.0, 0.0, 1.0))

		node = make_cloud_node(generate_torus_cloud(n=2_000, noise_st_dev=0.0))
		nodePath = NodePath(node, Thread.getCurrentThread())
		nodePath.reparentTo(self.render)
		nodePath.setPos(0, 0, 0)
		nodePath.setRenderModeThickness(0.15)
		nodePath.setRenderModePerspective(True)

		self.taskMgr.add(self.holdCamera, "HoldCameraTask")

	def holdCamera(self, task):
		angleDegrees = task.time * 10
		angleRadians = angleDegrees * (pi / 180.0)
		rad = 100

		self.camera.setPos(0, -rad*cos(angleRadians), -rad*sin(angleRadians))
		self.camera.setHpr(0, angleDegrees, task.time*3)
		return Task.cont

app = MyApp()
app.run()