import random

import cv2
import numpy as np

from mcmc import MetropolisHastings, AdaptiveMetropolisHastings
from rectangle import RectangleFit

def infer(mcmc_cls, model, n, show_interval=500, burnin=1000):
	mcmc = mcmc_cls(model.log_p, model.x0(), sigma=model.dim_scale(), use_log=True)
	for i in range(burnin):
		mcmc.sample()
	samples = []
	for i in range(n):
		samples.append(mcmc.sample(index=random.randrange(model.n)))
		if len(samples) % show_interval == 0:
			print('a:', mcmc.acceptance_rate(), 'sigma:', mcmc.sigma_scale, 'd:', model.distance(samples[-1])[0])
			image, penalty = model.make_image(samples[-1])
			cv2.imshow('%d steps' % len(samples), np.hstack((image, model.image)))
			cv2.waitKey(0)
			cv2.destroyAllWindows()

if __name__ == '__main__':
	image = cv2.imread('images/starry_small.jpg')
	image = cv2.imread('images/beach.png')

	# model = RectangleFit(8, image)
	# mcmc = AdaptiveMetropolisHastings(model.p, model.x0())
	# for i in range(1000):
		# mcmc.sample()
	# image, _ = model.make_image(mcmc.sample())

	model = RectangleFit(8, image)
	infer(AdaptiveMetropolisHastings, model, n=5000)
