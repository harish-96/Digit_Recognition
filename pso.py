import numpy as np


class Particle:
    """A single particle in the swarm"""

    def __init__(self, dimension, X_max=5, X_min=-5):
        self.position = []
        self.velocity = []
        for d in dimension:
            self.position.append(((X_max - X_min) *
                                  np.random.rand(d) + X_min))

            self.velocity.append((0.1 * (X_max - X_min) * np.random.rand(d) +
                                  0.1 * X_min))

        self.position = np.array(self.position)
        self.velocity = np.array(self.velocity)
        self.pbest = self.position

    def update_pbest(self, error_function):
        if error_function(self.position) < error_function(self.pbest):
            self.pbest = self.position
