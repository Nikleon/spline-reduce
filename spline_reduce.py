import sys, os
import random
from math import sin, cos, sqrt, exp, pi, inf
from itertools import tee

import cv2
import numpy as np
import scipy
import scipy.interpolate as si
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, SpectralClustering

import pygame
from pygame.locals import *

WIN_SIZE = (600, 600)
SUB_DIVS = (4, 4, 30)
SLICE_DIVS = (8, 8, 1)

IMG_PATH = ("hiker.jpg", "chess.jpg")[1]

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

def generate_noisy_sphere(
		n			= 10_000,
		center		= np.array([0, 0, 0]),
		radius		= 25,
		noise_st_dev= 0.5):
	points = []
	for i in range(n):
		new_pt = np.random.normal(loc=center, scale=noise_st_dev , size=(3, ))
		points.append(
			tuple((
				radius * new_pt / np.linalg.norm(new_pt)
				+ np.random.normal(loc=center, scale=noise_st_dev , size=(3, ))
			).tolist())
		)
	return points

def sigmoid(x):
	return 1 / (1 + exp(-x))

def bounding_box(pts):
	bounds = list()
	for dim in range(len(pts[0])):
		l_bound = pts[0][dim]
		r_bound = pts[0][dim]
		for pt in pts[1:]:
			l_bound = min(l_bound, pt[dim])
			r_bound = max(r_bound, pt[dim])
		bounds.append(
			(l_bound, r_bound)
		)
	return bounds

def pt_to_bucket_ixs(pt, sub_divs, bounds):
	ixs = list()
	for dim in range(len(pt)):
		lower, upper = bounds[dim]
		ixs.append(int(
			min(sub_divs[dim] * (pt[dim] - lower) / (upper - lower), sub_divs[dim] - 1)
		))
	return ixs

def bucket_ix_to_bounds(ixs, sub_divs, bounds):
	bounds = list()
	for dim in range(len(ixs)):
		lower, upper = bounds[dim]
		interval = (upper - lower) / sub_divs[dim]
		l_bound = lower + interval * ixs[dim]
		bounds.append(
			(l_bound, l_bound + interval)
		)
	return bounds

def bucket_ixs_to_cpt(ixs, sub_divs, bounds):
	pt = list()
	for dim in range(len(ixs)):
		lower, upper = bounds[dim]
		interval = (upper - lower) / sub_divs[dim]
		pt.append(
			lower + interval * (ixs[dim] + 0.5)
		)
	return tuple(pt)

# Note! [z, y, x] format
def bucket(pts, sub_divs):
	bounds = bounding_box(pts)
	buckets = [[[
		list()
		for x in range(sub_divs[0])]
		for y in range(sub_divs[1])]
		for z in range(sub_divs[2])]
	for pt in pts:
		ixs = pt_to_bucket_ixs(pt, sub_divs, bounds)
		buckets[ixs[2]][ixs[1]][ixs[0]].append(pt)
	return buckets

def gen_chunks(sub_divs):
	for z in range(sub_divs[2]):
		for y in range(sub_divs[1]):
			for x in range(sub_divs[0]):
				yield (x, y, z)

def slice_ixs(xs=[], ys=[], zs=[]):
	out = []
	for (x, y, z) in gen_chunks(SUB_DIVS):
		if (x in xs) or (y in ys) or (z in zs):
			out.append( (x, y, z) )
	return out

def slice_buckets(buckets, xs=[], ys=[], zs=[]):
	ixss = slice_ixs(xs, ys, zs)
	out = []
	for ixs in ixss:
		out.extend(buckets[ixs[2]][ixs[1]][ixs[0]])
	return out

# Source: https://github.com/ciortanmadalina/high_noise_clustering/blob/master/spectral_clustering.ipynb
def affinity_matrix(X, k=7):
	# X: (m observations, n dims)
	dists = squareform(pdist(X))

	knn_dists = np.sort(dists, axis=0)[min(k, len(dists)-1)]
	knn_dists = knn_dists[np.newaxis].T

	local_scale = knn_dists.dot(knn_dists.T)

	aff_matrix = dists * dists
	aff_matrix = -aff_matrix / local_scale

	aff_matrix[np.where(np.isnan(aff_matrix))] = 0.0

	aff_matrix = np.exp(aff_matrix)
	np.fill_diagonal(aff_matrix, 0)
	return aff_matrix

def eigen_decomp(A):
	L = csgraph.laplacian(A, normed=True)
	n_components = A.shape[0]

	eigenvals, eigenvecs = eigsh(L, k=n_components, which="LM", sigma=1.0)

	num_clusters_optimal = np.argmax(np.diff(eigenvals)[:2]) + 1

	return num_clusters_optimal, eigenvecs

def eigen_knn(num_clusters, X):
	sc = SpectralClustering(
		n_clusters	= num_clusters,
		gamma		= 0.1
	)
	sc.fit(X)
	return sc.labels_

def bucket_averages(buckets, sub_divs):
	# Avgs to initialize
	avgs = [[[
		set()
		for x in range(sub_divs[0])]
		for y in range(sub_divs[1])]
		for z in range(sub_divs[2])]
	max_density = 0
	for (x, y, z) in gen_chunks(sub_divs):
		chunk = buckets[z][y][x]
		if len(chunk) > 0:
			val = tuple(map(np.mean, zip(*chunk)))
			avgs[z][y][x] = val
			max_density = max(max_density, len(chunk))
		else:
			avgs[z][y][x] = None
	for (x, y, z) in gen_chunks(sub_divs):
		if len(buckets[z][y][x]) < 0.5 * max_density:
			avgs[z][y][x] = None
	return avgs

# Source: https://stackoverflow.com/questions/24612626/b-spline-interpolation-with-python
def bspline(cv, n=50, degree=3, periodic=True):
	cv = np.asarray(cv)
	count = len(cv)

	if periodic:
		factor, fraction = divmod(count+degree+1, count)
		cv = np.concatenate((cv,) * factor + (cv[:fraction],))
		count = len(cv)
		degree = np.clip(degree,1,degree)
	else:
		degree = np.clip(degree,1,count-1)

	kv = None
	if periodic:
		kv = np.arange(0-degree,count+degree+degree-1, dtype='int')
	else:
		kv = np.concatenate(([0]*degree, np.arange(count-degree+1), [count-degree]*degree))

	u = np.linspace(periodic,(count-degree),n)

	return (np.array(si.splev(u, (kv,cv.T,degree))).T, 
			np.array(si.splev(u, (kv,cv.T,degree), der=1)).T,
			u)

def algorithm(cities):
	best_order = []
	best_length = float('inf')

	for i_start, start in enumerate(cities):
		order = [i_start]
		length = 0

		i_next, next, dist = get_closest(start, cities, order)
		length += dist
		order.append(i_next)

		while len(order) < cities.shape[0]:
			i_next, next, dist = get_closest(next, cities, order)
			length += dist
			order.append(i_next)

		if length < best_length:
			best_length = length
			best_order = order
			
	return best_order, best_length

def get_closest(city, cities, visited):
	best_distance = float('inf')

	for i, c in enumerate(cities):

		if i not in visited:
			distance = dist_squared(city, c)

			if distance < best_distance:
				closest_city = c
				i_closest_city = i
				best_distance = distance

	return i_closest_city, closest_city, best_distance

def dist_squared(c1, c2):
	t1 = c2[0] - c1[0]
	t2 = c2[1] - c1[1]

	return t1**2 + t2**2

def sdm(pts):
	rand_order = list(range(len(pts)))
	random.shuffle(rand_order)
	avg_list = [pts[rand_order[i]] for i in range(20)
				if i < len(rand_order)
					and pts[rand_order[i]] is not None]
	ix_order, _ = algorithm(np.array(avg_list))
	cv = [avg_list[ix_order[i]] for i in range(len(ix_order))]

	lm = 0.01
	for i in range(20):
		P, Pprime, ts = bspline(cv, n=100)
		normals = [-dx/dy for (dx, dy, _) in Pprime]

		f_SD_sum = 0
		f_s_sum = 0
		for k in range(len(pts)):
			(x0, y0, _) = pts[k]

			ix1 = 0
			min_d1 = inf
			ix2 = 0
			min_d2 = inf
			for j in range(len(P)):
				(x1, y1, _) = P[j]

				v = (1, normals[j]) # TODO maybe swap
				r = (x1 - x0, y1 - y0)

				v_norm = sqrt( v[0]**2 + v[1]**2 )
				v_unit = (v[0] / v_norm, v[1] / v_norm)

				d = abs( v_unit[0] * r[0] + v_unit[1] * r[1] )

				if d < min_d1:
					ix2 = ix1
					min_d2 = min_d1
					ix1 = j
					min_d1 = d
				elif d < min_d2:
					ix2 = j
					min_d2 = d

			t1 = ts[ix1]
			t2 = ts[ix2]
			t_approx = (min_d2 * t1 + min_d1 * t2) / (min_d1 + min_d2)
			d_approx = (min_d1 + min_d2) / 2 #TODO maybe toooo simple

			#e_SD_k = #TODODODODODODODO
			#f_SD_sum += e_SD_k()

			# compute SD error term f_SD
			# solve linsys to minimize f_SD

			#f_s_sum += Pplus ** 2
		f_SD_sum /= 2
		f_SD_sum += lm * f_s_sum

	return cv



class PointCloud:
	'''
	Replication Pipeline (CVP)
	---------------------------=============================

	Semiglobal Matching			Gain depth info from dual-cam setup

	Watershed Segmentation		Isolate target object

	Eigengap heuristic			Predict topology of scanned piece

	Spectral Clustering			Identify physically disjunct cross-sectional areas

	Greedy-TSP warm-start		Provide initial approx. of cs-area edges

	Quadtree preprocessing		Increase SDM principal point projection efficiency

	SDM B-Spline Fitting		Make robust to noise

	Quantization				Toggle level of detail

	Lossless Compression		Prepare to be transmitted
	'''

	def __init__(self, pts=None):
		if pts is None:
			self.set_points( generate_noisy_sphere() )
		else:
			self.set_points( pts )
		self.screen = None

	def set_points(self, pts):
		self.points = pts
		self.buckets = bucket(pts, SUB_DIVS)
		self.hl = []
		self.hl_cluster = []

	def highlight(self, xs=[], ys=[], zs=[]):
		self.hl = slice_ixs(xs, ys, zs)

	def draw(self):
		if self.screen is None:
			pygame.init()
			self.screen = pygame.display.set_mode(WIN_SIZE, pygame.NOFRAME)
			#pygame.display.set_caption('Craitor - spline_reduce')

		spread = 10
		min_w = 2
		k_w = 4
		min_r = 150
		max_r = 200

		self.screen.fill(color=(180, 180, 180))
		for (x, y, z) in gen_chunks(SUB_DIVS):
			chunk = self.buckets[z][y][x]
			if len(chunk) > 0:
				for pos in chunk:
					if pos in self.hl_cluster:
						col = (10, min_r + int( (max_r - min_r)
								* sigmoid(pos[2]) ), 10)
					elif (x, y, z) in self.hl:
						col = (255, 255, 255)
					else:
						if len(self.hl_cluster) > 0:
							pass #DEBUG
						col = (
							min_r+int((max_r - min_r)*sigmoid(pos[2])),
							10,
							10
						)

					pygame.draw.circle(
						self.screen,
						col,
						(
							int(WIN_SIZE[0] / 2 + spread * pos[0]),
							int(WIN_SIZE[1] / 2 + spread * pos[1])
						),
						max(min_w, int(k_w * sigmoid(pos[2])))
					)
		pygame.display.update()

	def render2d_hook(self):
		z = 1
		self.draw()
		pygame.event.clear()
		while True:
			event = pygame.event.wait()
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
			elif event.type == KEYDOWN:
				if event.key == K_ESCAPE or event.key == K_q:
					pygame.quit()
					sys.exit()
				elif event.key == K_t:
					self.set_points(generate_torus_cloud())
					z=1
				elif event.key == K_s:
					self.set_points(generate_noisy_sphere())
					z=1
				elif event.key == K_0:
					self.highlight(zs=[0])
				elif event.key == K_1:
					self.highlight(zs=[1])
				elif event.key == K_2:
					self.highlight(zs=[2])
				elif event.key == K_3:
					self.highlight(zs=[3])
				elif event.key == K_4:
					self.highlight(zs=[4])
				elif event.key == K_l:
					if z < SUB_DIVS[2]:
						bucket_slice = slice_buckets(self.buckets, zs=[z])
						bs = [pt[:2] for pt in bucket_slice]
						X = np.array(bs)
						if len(X) < 3:
							continue
						n, _ = eigen_decomp(affinity_matrix(X))
						labels = eigen_knn(n, X)

						clusters = [list() for _ in range(n)]
						for i in range(len(labels)):
							clusters[labels[i]].append(bucket_slice[i])
						for cluster in clusters:
							#self.draw()
							'''
							cluster_buckets = bucket(cluster, SLICE_DIVS)
							avgs = bucket_averages(cluster_buckets, SLICE_DIVS)
							avg_list = list()
							for (x, y, z) in gen_chunks(SLICE_DIVS):
								a = avgs[z][y][x]
								if not a is None:
									cpt = bucket_ixs_to_cpt((x, y, z), SLICE_DIVS, bounding_box(cluster))
									pygame.draw.circle(self.screen, (10, 10, 200), (
										int(WIN_SIZE[0] / 2 + 10 * cpt[0]),
										int(WIN_SIZE[1] / 2 + 10 * cpt[1])
									), 5)
									avg_list.append(a)
							'''
							rand_order = list(range(len(cluster)))
							random.shuffle(rand_order)
							avg_list = [cluster[rand_order[i]] for i in range(20)
										if i < len(rand_order)
											and cluster[rand_order[i]] is not None]
							if len(avg_list) >= 3:
								ix_order, _ = algorithm(np.array(avg_list))
								cv = [avg_list[ix_order[i]] for i in range(len(ix_order))]
								#cv = sdm(cluster)
								samples, _, _ = bspline(cv, 100)
								ss = [(np.array(WIN_SIZE) / 2 + 10*pt[:2]).tolist() for pt in samples]
								sc = 255 / SUB_DIVS[2]
								pygame.draw.polygon(self.screen, (min(sc*z, 255), min(sc*z, 255), min(sc*z, 255)), ss)
								#pygame.draw.lines(self.screen, (10, 10, 200), True, ss, 5)
								pygame.display.update()
						z += 1
						continue
					else:
						continue
				self.draw()

'''
from PIL import Image
im = Image.open("hiker.jpg")
np_im = np.array(im).mean(axis=2)
w, h = np_im.shape
s = 0.1

thresh = 50
pts = list()
for x in range(w):
	for y in range(h):
		dp = np_im[x][y]
		if dp >= thresh:
			if random.uniform(0, 1) < 0.02:
				while dp >= thresh:
					r = np.random.normal(scale=0.1 , size=(3, ))
					pts.append( (s * (y - h/2) + r[0]-7, s * (x - w/2) + r[1]-3, 10*(dp - (255 - thresh)/2)) )
					dp -= 29

p = PointCloud(pts)
p.render2d_hook()
'''

from PIL import Image
img_grayscale = np.array(Image.open(IMG_PATH)).mean(axis=2)
img_grayscale_float = img_grayscale / np.max(img_grayscale)

h, w = img_grayscale_float.shape
print(img_grayscale_float.shape)
d = max(w, h)
scale = 100
pts = []
for y in range(h):
	for x in range(w):
		if random.uniform(0, 1) > 1:
			continue
		gray = img_grayscale_float[(h-1)-y][x]
		pts.append((scale * (x - w/2) / d, scale * (0.5 - gray), scale * (y - h/2) / d))

#NEW START
from cloud_render import *
app = MyApp()

#pts = generate_torus_cloud(n=10_000, noise_st_dev=0.1)
app.render_cloud(pts)

#p = PointCloud(pts)
#p.render2d_hook()

app.run()




