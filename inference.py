import random

import cv2
import numpy as np

from mcmc import MetropolisHastings, AdaptiveMetropolisHastings
from rectangle import RectangleFit

def infer(mcmc_cls, model, n, show_interval=500, burnin=0, sigma_scale=1, saveas=None):
	if show_interval is None:
		show_interval = n
	sigma = [sigma_scale*dim for dim in model.dim_scale()]
	mcmc = mcmc_cls(model.log_p, model.x0(), sigma=sigma, use_log=True)
	# for i in range(burnin):
		# mcmc.sample()
	samples = []
	for i in range(n):
		samples.append(mcmc.sample(index=random.randrange(model.n)))
		if len(samples) % show_interval == 0:
			print('steps:', len(samples), 'a:', mcmc.acceptance_rate(), 'sigma:', sigma_scale*mcmc.sigma_scale, 'd:', model.distance(samples[-1])[0])
			image, penalty = model.make_image(samples[-1])
			if saveas is not None:
				cv2.imwrite('generated/%s_%d.png' % (saveas, len(samples)), image)
			else:
				cv2.imshow('%d steps' % len(samples), np.hstack((image, model.image)))
				cv2.waitKey(0)
				cv2.destroyAllWindows()

if __name__ == '__main__':
	image = cv2.imread('images/rectangles.png')
	image = cv2.imread('images/supper.jpg')
	image = cv2.imread('images/beach.png')
	image = cv2.imread('images/blue.png')
	image = cv2.imread('images/adam.jpg')
	image = cv2.imread('images/sunday.jpg')
	image = cv2.imread('images/starry.jpg')

	# model = RectangleFit(8, image)
	# mcmc = AdaptiveMetropolisHastings(model.log_p, model.x0(), use_log=True)
	# for i in range(1000):
		# mcmc.sample()
	# image, _ = model.make_image(mcmc.sample())

	image_names = [
	'rectangles',
	# 'supper',
	'beach',
	'blue',
	# 'adam',
	'sunday',
	'starry',
	]
	for image_name in image_names:
		image = cv2.imread('images/%s.png' % image_name)
		if image is None:
			image = cv2.imread('images/%s.jpg' % image_name)
		assert image is not None
		model = RectangleFit(8, image)
		infer(AdaptiveMetropolisHastings, model, n=5000, sigma_scale=0.05, saveas=image_name+'_adaptive')
		infer(MetropolisHastings, model, n=5000, sigma_scale=0.05, saveas=image_name)
