import random

import cv2
import numpy as np

from mcmc import MetropolisHastings, AdaptiveMetropolisHastings
from rectangle import RectangleFit

def infer(mcmc_cls, model, n, show_interval=500, burnin=1000, sigma_scale=1):
	sigma = [sigma_scale*dim for dim in model.dim_scale()]
	mcmc = mcmc_cls(model.log_p, model.x0(), sigma=sigma, use_log=True)
	for i in range(burnin):
		mcmc.sample()
	samples = []
	for i in range(n):
		samples.append(mcmc.sample(index=random.randrange(model.n)))
		if len(samples) % show_interval == 0:
			print('steps:', len(samples), 'a:', mcmc.acceptance_rate(), 'sigma:', sigma_scale*mcmc.sigma_scale, 'd:', model.distance(samples[-1])[0])
			image, penalty = model.make_image(samples[-1])
			cv2.imshow('%d steps' % len(samples), np.hstack((image, model.image)))
			cv2.waitKey(0)
			cv2.destroyAllWindows()

if __name__ == '__main__':
	image = cv2.imread('images/rectangles.png')
	image = cv2.imread('images/supper.jpg')
	image = cv2.imread('images/beach.png')
	image = cv2.imread('images/starry.jpg')
	image = cv2.imread('images/starry_small.jpg')
	image = cv2.imread('images/sunday.jpg')
	image = cv2.imread('images/blue.png')
	image = cv2.imread('images/adam.jpg')

	# model = RectangleFit(8, image)
	# mcmc = AdaptiveMetropolisHastings(model.log_p, model.x0(), use_log=True)
	# for i in range(1000):
		# mcmc.sample()
	# image, _ = model.make_image(mcmc.sample())

	model = RectangleFit(8, image)
	infer(AdaptiveMetropolisHastings, model, n=5000, sigma_scale=0.5)
