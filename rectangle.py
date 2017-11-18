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
			tuple(np.array([0.02, 0.125]) * np.sum(self.image.shape[:2])), # line width
			(0.1, 0.5), # opacity
		]

	def dim_scale(self):
		return [x_max - x_min for x_min, x_max in self.range()]

	def distance(self, x):
		generated, penalty = self.make_image(x)
		distance = np.mean(np.abs(generated - self.image))
		return distance, penalty / self.n

	def log_p(self, x):
		c_distance = 2
		c_penalty = 100
		distance, penalty = self.distance(x)
		score = -c_distance*distance-c_penalty*penalty
		return score

	def make_image(self, x):
		image = np.full_like(self.image, 255)
		penalty_total = 0
		for rect in x:
			image, penalty = self.draw(image, rect)
			penalty_total += penalty
		return image, penalty_total

	def draw(self, image, rect):
		penalty_total = 0

		rect_clipped = []
		for val, bounds in zip(rect, self.range()):
			val, penalty = self.clip(val, *bounds)
			rect_clipped.append(val)
			penalty_total += penalty
		pt0, pt1, color, width, opacity = rect_clipped

		# color = cv2.cvtColor(color.astype(np.uint8)[None,None,...], cv2.COLOR_HSV2BGR)[0,0].astype(float)
		pt0 = (pt0 * (1 << self.BITSHIFT)).astype(int)
		pt1 = (pt1 * (1 << self.BITSHIFT)).astype(int)
		width = int(width[0])
		opacity = opacity[0]

		image = image.copy()
		overlay = image.copy()
		cv2.line(overlay, tuple(pt0), tuple(pt1), tuple(color), width, lineType=cv2.LINE_AA, shift=self.BITSHIFT)
		return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0), penalty_total

	@staticmethod
	def clip(a, a_min, a_max):
		penalty = np.sum(np.maximum(0, np.maximum(a_min - a, a - a_max)) / (a_max - a_min))
		return np.clip(a, a_min, a_max), penalty

if __name__ == '__main__':
	image = cv2.imread('images/starry.jpg')
	fit = RectangleFit(8, image)
	img, _ = fit.make_image([
		[
			np.array([0,0], dtype=float), # pt0
			np.array([200,200], dtype=float), # pt1
			np.array([255,55,55], dtype=float), # color
			np.array([5]), # line width
			np.array([0.5]), # opacity
		],
		[
			np.array([80,40], dtype=float), # pt0
			np.array([20,200], dtype=float), # pt1
			np.array([55,55,255], dtype=float), # color
			np.array([40]), # line width
			np.array([0.5]), # opacity
		],
	])
	cv2.imshow('', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
