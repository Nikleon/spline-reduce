from math import sin, cos, pi
from numpy import linspace
import sys

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from panda3d.core import Thread, LColorf
from panda3d.core import GeomVertexData, GeomVertexFormat, GeomVertexWriter, GeomEnums
from panda3d.core import NodePath, GeomNode, Geom, RenderState, GeomPrimitive, GeomPoints


def make_cloud_node(pts, col=LColorf(1.0, 0.0, 0.0, 1.0)):
	ptCloudData = GeomVertexData("PointCloudData", GeomVertexFormat.getV3c4(), GeomEnums.UH_static)
	vertexWriter = GeomVertexWriter(ptCloudData, Thread.getCurrentThread())
	vertexWriter.setColumn("vertex")
	colorWriter = GeomVertexWriter(ptCloudData, Thread.getCurrentThread())
	colorWriter.setColumn("color")

	for (x, y, z) in pts:
		vertexWriter.addData3(x, y, z)
		colorWriter.addData4(col)

	geomPts = GeomPoints(GeomEnums.UH_static)
	geomPts.addConsecutiveVertices(0, len(pts))
	geomPts.closePrimitive()

	geom = Geom(ptCloudData)
	geom.addPrimitive(geomPts)

	node = GeomNode("PointCloudNode")
	node.addGeom(geom, RenderState.makeEmpty())
	return node

def make_patch_node(pts, col=LColorf(0.0, 1.0, 0.0, 1.0)):
	splinePatchData = GeomVertexData("SplinePatchData", GeomVertexFormat.getV3c4(), GeomEnums.UH_static)
	vertexWriter = GeomVertexWriter(splinePatchData, Thread.getCurrentThread())
	vertexWriter.setColumn("vertex")
	colorWriter = GeomVertexWriter(splinePatchData, Thread.getCurrentThread())
	colorWriter.setColumn("color")

	for (x, y, z) in pts:
		vertexWriter.addData3(x, y, z)
		colorWriter.addData4(col)

	geomLines = GeomLines(GeomEnums.UH_static)
	geomLines.addConsecutiveVertices(0, len(pts))
	geomLines.addVertex(0)
	geomLines.closePrimitive()

	geom = Geom(splinePatchData)
	geom.addPrimitive(geomLines)

	node = GeomNode("SplinePatchNode")
	node.addGeom(geom, RenderState.makeEmpty())
	return node


class MyApp(ShowBase):

	def __init__(self):
		ShowBase.__init__(self)

		self.win.setClearColorActive(True)
		self.win.setClearColor(LColorf(0.0, 0.0, 0.0, 1.0))

		self.animate = False
		self.taskMgr.add(self.animateCamera, "AnimateCameraTask")

		#base.messenger.toggleVerbose()
		self.accept("`", self.toggleAnimate)
		self.accept("q", sys.exit)
		self.accept('escape', sys.exit)

	def toggleAnimate(self):
		self.animate = not self.animate

	def animateCamera(self, task):
		radius = 200
		if self.animate:
			angleDegrees = task.time * 10
			angleRadians = angleDegrees * (pi / 180.0)
			self.camera.setPos(0, -radius * cos(angleRadians), -radius * sin(angleRadians))
			self.camera.setHpr(0, angleDegrees, task.time * 3)
		return Task.cont

	def render_cloud(self, pts):
		nodePath = NodePath(make_cloud_node(pts), Thread.getCurrentThread())
		nodePath.reparentTo(self.render)

		nodePath.setRenderModeThickness(0.15)
		nodePath.setRenderModePerspective(False)

	def render_patch(self, pts):
		nodePath = NodePath(make_patch_node(pts), Thread.getCurrentThread())
		nodePath.reparentTo(self.render)

		nodePath.setRenderModeThickness(0.2)
		nodePath.setRenderModePerspective(True)

	def render_spline(self, ctrs):
		bounds = (0, 1)
		numPts = 100
		pts = [eval_b_spline(ctrs, t) for t in linspace(bounds[0], bounds[1], numPts)]
		self.render_patch(pts)
