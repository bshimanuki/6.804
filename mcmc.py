import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np

class MetropolisHastings(object):
	def __init__(self, p, x0, q=None, sigma=1, use_log=False):
		if q is None:
			self.q = self.proposal
		else:
			self.q = q
		self.p = p
		self.x = x0
		self.sigma = sigma
		self.sigma_scale = 1
		self.use_log = use_log
		self.accepted = 0
		self.rejected = 0

	def accept(self, x):
		if self.use_log:
			alpha = math.exp(self.p(x) - self.p(self.x))
		else:
			alpha = self.p(x) / self.p(self.x)
		if alpha > random.random():
			self.x = x
			self.accepted += 1
			return True
		else:
			self.rejected += 1
			return False

	def proposal(self, x, sigma=None):
		if isinstance(x, list):
			if x and not isinstance(x[0], list) and isinstance(self.sigma, list):
				return [self.proposal(xi, sigma=s) for xi, s in zip(x, self.sigma)]
			return [self.proposal(xi) for xi in x]
		if sigma is None:
			sigma = self.sigma
		return np.random.normal(loc=x, scale=sigma)

	def sample(self, index=None):
		if isinstance(self.x, list) and index is not None:
			x = copy.deepcopy(self.x)
			x[index] = self.q(self.x[index])
		else:
			x = self.q(self.x)
		self.accept(x)
		return self.x

	def acceptance_rate(self):
		return self.accepted / (self.accepted + self.rejected)

class AdaptiveMetropolisHastings(MetropolisHastings):
	def __init__(self, p, x0, sigma=1, use_log=False, halflife=500):
		super().__init__(p, x0, q=self.proposal, sigma=sigma, use_log=use_log)
		self.target_acceptance = 0.234 # optimal acceptance rate as d -> infinity
		# self.target_acceptance = 0.44 # optimal acceptance rate for d=1
		self.factor = 1
		self.factor_halflife = halflife

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

def visualize(cls, p, x0, n, sigma=None, f_range=None, burnin=1000, num_bins=20, title=None, save=None):
	'''Visualization for a 1d state space.'''
	if sigma is None:
		mcmc = cls(p, x0)
	else:
		mcmc = cls(p, x0, sigma=sigma)
	for i in range(burnin):
		mcmc.sample()
	samples = []
	for i in range(n):
		samples.append(mcmc.sample())
	plt.hist(samples, bins=num_bins, normed=1, alpha=0.5)
	plt.xlabel('$x$')
	plt.ylabel('$P(x)$')
	if title is not None:
		plt.title(title)
	if f_range is not None:
		x_min, x_max = f_range
		dx = (x_max-x_min)/100
		x = np.arange(x_min, x_max, dx)
		y = p(x)
		y = y / np.sum(y) / dx
		plt.plot(x, y)
	if save is None:
		print(mcmc.acceptance_rate())
		plt.show()
	else:
		plt.savefig(save)
	plt.clf()

def make_peaks(l):
	def f(x):
		return np.maximum(0, np.minimum(abs(x), l-abs(x)))
	return f

def compare(p, sigmas, name, title, f_range=None, x0=0.1, n=1000):
	for sigma in sigmas:
		visualize(
			MetropolisHastings, p, x0, n, f_range=f_range, sigma=sigma,
			save='%s_%s.png'%(name, sigma),
			title='Nonadaptive %s ($\sigma=%s$)'%(title, sigma),
		)
	visualize(
		AdaptiveMetropolisHastings, p, x0, n, f_range=f_range,
		save='%s_adaptive.png'%name,
		title='Adaptive %s'%title,
	)

if __name__ == '__main__':
	p = lambda x: max(0, 10-x**2)
	p = lambda x: math.exp(-x**2/2)
	p = make_peaks(20)
	ls = [1, 10, 100]
	for l in ls:
		name = 'plots/doublepeak%d' % l
		title = 'Size %d Double Peaks' % l
		compare(make_peaks(l), [0.1, 1, 10], name, title, f_range=(-l,l))
