
import pygame
from pygame.locals import *

import numpy as np

from scipy.interpolate import BSpline

# Note: order gives information, each byte could subdivide only to one side of the previous rerefence points

def sample_bspline(
	cv		= [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5],
	knots	= [0, 0, 0, 0, 1, 0, 0, 0, 0],
	degree	= 3):

	k = 2
	t = [0, 1, 2, 3, 4, 5, 6]
	c = [-1, 2, 0, -1]
	spl = BSpline(t, c, k)
	pts = spl(np.linspace(cv[0], cv[-1], 100))
	
	return pts


class GUI:
	WIN_SIZE	= (600, 600)
	TITLE		= "Footpoint investigation"

	def __init__(self):
		pygame.init()
		pygame.display.set_caption(GUI.TITLE)
		self.screen = None

	def toggle_screen(self):
		if (self.screen is None) or self.is_fullscreen:
			self.screen = pygame.display.set_mode(GUI.WIN_SIZE)
			self.is_fullscreen = False
		else:
			modes = pygame.display.list_modes()
			if len(modes) == 0:
				return
			self.screen = pygame.display.set_mode(modes[0], FULLSCREEN)
			self.is_fullscreen = True

	def _yield_events(self):
		pygame.event.clear()
		while not self.dispose_requested:
			yield pygame.event.wait()

	def render_hook(self):
		for evt in self._yield_events():
			if evt.type == QUIT:
				self.dispose_requested = True

			elif evt.type == KEYDOWN:
				if evt.key == K_ESCAPE or evt.key == K_q:
					self.dispose_requested = True

				elif evt.key == K_BACKQUOTE:

					pts = [(1, 2), (3, 4), (50, 50)]
					pygame.draw.aalines(g.screen, Color("green"), True, pts)
					pygame.display.update()

				elif evt.key == K_f:
					self.toggle_screen()

	def show(self):
		if self.screen is None:
			self.toggle_screen()
		self.dispose_requested = False

print(sample_bspline())

g = GUI()
g.show()
g.render_hook()