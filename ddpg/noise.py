""" noise.py contains class(es) used for adding noise to data. Currently
    contains the lone class OrnsteinUhlenbeckActionNoise, which is used to add
    Ornstein Uhlenbeck noise to the actor policy in DDPG.

    Author: Jonathon Sather
    Last updated: 2/18/2018

    Largely copied from:
    https://github.com/pemami4911/deep-rl/tree/master/ddpg
    ... who took it from:
    https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    ... which was based on:
    http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
"""
import numpy as np

class OrnsteinUhlenbeckActionNoise:
    """ Defines a noise object used for adding Ornstein Uhlenbeck noise to data.
    """
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        """ Initialize the noise parameters and reset the memory. """

        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """ Calculate and return a noise value. """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """ Reset the memory. """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        """ Return diagnoistic information for data logging. """
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
