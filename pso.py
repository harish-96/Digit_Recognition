import numpy as np


class Particle:
    """A single particle in the swarm. Represents one possible solution
    of the optimisation problem. Initialise with dimension of the design space
    and bounds for the variables"""

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
        """Updates the best position of the particle in its own
        personal history"""
        if error_function(self.position) < error_function(self.pbest):
            self.pbest = self.position


class Swarm(Particle):
    """A swarm of particles which move around to find the optimum in the design
    space. Initialise with number of particles in the swarm, error function to
    compute fitness. Probability of death of particles - which if higher leads
    to more exploration and exploitation, the parameters of PSO - w, c1 and c2
    and exit error which is the error value at which the optimisation
    procedure concludes"""

    def __init__(self, numParticles, error_function, pdeath=0.005,
                 dimension=[784 * 15, 15 * 10, 15, 10], w=0.729, c1=1.49445,
                 c2=1.49445, exit_error=0.1):
        self.dimension = dimension
        self.exit_error = exit_error
        self.error_function = error_function
        self.swarm = np.asarray([Particle(self.dimension)
                                 for i in range(numParticles)])
        gbest_index = np.argmin([self.error_function(self.swarm[i].position)
                                 for i in range(numParticles)])
        self.gbest = self.swarm[gbest_index].position
        self.numParticles = numParticles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pdeath = pdeath

    def update_gbest(self):
        """Update the current best position of the swarm based on the fitness
        function defined in the init function"""
        gbest_index = np.argmin([self.error_function(self.swarm[i].position)
                                 for i in range(self.numParticles)])
        self.gbest = self.swarm[gbest_index].position
