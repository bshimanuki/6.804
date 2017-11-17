import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

class RectangleFit(object):
	BITSHIFT = 16 # subpixel precision

	def __init__(self, n, image):
		self.image = image
		self.n = n

	def x0(self):
		return [
			[
				x_min + np.random.random(num_param) * (x_max - x_min)
				for num_param, (x_min, x_max) in zip(self.num_params(), self.range())
			]
			for i in range(self.n)]

	def num_params(self):
		return [
			2, # pt0
			2, # pt1
			3, # color
			1, # line width
			1, # opacity
		]

	def range(self):
		return [
			(0, np.array(self.image.shape[:2])), # pt0
			(0, np.array(self.image.shape[:2])), # pt1
			(0, 256), # color
			(2, np.sum(self.image.shape[:2])), # line width
			(0.1, 0.75), # opacity
		]

	def dim_scale(self):
		return [x_max - x_min for x_min, x_max in self.range()]

	def distance(self, x):
		generated, valid = self.make_image(x)
		distance = np.mean(np.abs(generated - self.image))
		return distance, valid

	def p(self, x):
		distance, valid = self.distance(x)
		score = math.exp(-distance)
		return score if valid else 0

	def make_image(self, x):
		image = np.full_like(self.image, 255)
		valid_all = True
		for rect in x:
			image, valid = self.draw(image, rect)
			valid_all &= valid
		return image, valid

	def draw(self, image, rect):
		valid_all = True

		rect_clipped = []
		for val, bounds in zip(rect, self.range()):
			val, valid = self.clip(val, *bounds)
			rect_clipped.append(val)
			valid_all &= valid
		pt0, pt1, color, width, opacity = rect_clipped

		pt0 = (pt0 * (1 << self.BITSHIFT)).astype(int)
		pt1 = (pt1 * (1 << self.BITSHIFT)).astype(int)
		width = int(width[0])
		opacity = opacity[0]

		image = image.copy()
		overlay = image.copy()
		cv2.line(overlay, tuple(pt0), tuple(pt1), tuple(color), width, lineType=cv2.LINE_AA, shift=self.BITSHIFT)
		return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0), valid_all

	@staticmethod
	def clip(a, a_min, a_max):
		valid = True
		if a_min is not None and np.any(a < a_min):
			valid = False
		if a_max is not None and np.any(a >= a_max):
			valid = False
		return np.clip(a, a_min, a_max), valid

if __name__ == '__main__':
	image = cv2.imread('images/starry.jpg')
	fit = RectangleFit(8, image)
	img = fit.make_image([
		[
			np.array([0,0], dtype=float), # pt0
			np.array([200,200], dtype=float), # pt1
			np.array([255,55,55], dtype=float), # color
			5, # line width
			0.5, # opacity
		],
		[
			np.array([80,40], dtype=float), # pt0
			np.array([20,200], dtype=float), # pt1
			np.array([55,55,255], dtype=float), # color
			40, # line width
			0.5, # opacity
		],
	])
	cv2.imshow('', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
