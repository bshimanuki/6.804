import math
import random

import matplotlib.pyplot as plt
import numpy as np

class MetropolisHastings(object):
	def __init__(self, p, x0, q=None, sigma=1):
		if q is None:
			def proposal(x):
				if isinstance(x, list):
					return [proposal(xi) for xi in x]
				return np.random.normal(loc=x, scale=sigma)
			self.q = proposal
		else:
			self.q = q
		self.p = p
		self.x = x0
		self.accepted = 0
		self.rejected = 0

	def accept(self, x):
		alpha = self.p(x) / self.p(self.x)
		if alpha > random.random():
			self.x = x
			self.accepted += 1
			return True
		else:
			self.rejected += 1
			return False

	def sample(self):
		x = self.q(self.x)
		self.accept(x)
		return self.x

	def acceptance_rate(self):
		return self.accepted / (self.accepted + self.rejected)

class AdaptiveMetropolisHastings(MetropolisHastings):
	def __init__(self, p, x0, sigma=1):
		super().__init__(p, x0, q=self.proposal)
		self.sigma = sigma
		self.sigma_scale = 1
		self.target_acceptance = 0.234 # optimal acceptance rate as d -> infinity
		# self.target_acceptance = 0.44 # optimal acceptance rate for d=1
		self.factor = 1
		self.factor_halflife = 1000

	def accept(self, x):
		val = super().accept(x)
		if val:
			self.sigma_scale *= math.exp((1 - self.target_acceptance) * self.factor)
			if isinstance(self.sigma, list):
				self.sigma = [s * math.exp((1 - self.target_acceptance) * self.factor) for s in self.sigma]
			else:
				self.sigma *= math.exp((1 - self.target_acceptance) * self.factor)
		else:
			self.sigma_scale /= math.exp(self.target_acceptance * self.factor)
			if isinstance(self.sigma, list):
				self.sigma = [s / math.exp(self.target_acceptance * self.factor) for s in self.sigma]
			else:
				self.sigma /= math.exp(self.target_acceptance * self.factor)
		self.factor *= (1/2) ** (1 / self.factor_halflife)
		return val

	def proposal(self, x, sigma=None):
		if isinstance(x, list):
			if isinstance(self.sigma, list):
				return [self.proposal(xi, sigma=s) for xi, s in zip(x, self.sigma)]
			return [self.proposal(xi) for xi in x]
		if sigma is None:
			sigma = self.sigma
		return np.random.normal(loc=x, scale=sigma)

def visualize(cls, p, x0, n, burnin=1000):
	'''Visualization for a 1d state space.'''
	mcmc = cls(p, x0)
	for i in range(burnin):
		mcmc.sample()
	samples = []
	for i in range(n):
		samples.append(mcmc.sample())
	plt.hist(samples, bins='auto')
	print(mcmc.acceptance_rate())
	plt.show()

if __name__ == '__main__':
	p = lambda x: max(0, 10-x**2)
	p = lambda x: math.exp(-x**2/2)
	p = lambda x: max(0, min(abs(x), 20-abs(x)))
	cls = MetropolisHastings
	visualize(cls, p, 0.1, 1000)
	cls = AdaptiveMetropolisHastings
	visualize(cls, p, 0.1, 1000)
