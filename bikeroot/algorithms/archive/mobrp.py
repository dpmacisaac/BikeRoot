from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import random
import numpy as np

class MOBRP(Problem):
    def __init__(self, graph, n_obj=4,seed=0):
        super().__init__(
            n_var=1,
            n_obj=n_obj,
            n_constr=0,
            xl=0.0,            # adjust if you use a real encoding
            xu=1.0,
            elementwise_evaluation=False,   # evaluate whole population at once
        )
        self.graph = graph
        self.route_storage = []
        self.rand = random.Random()
        self.rand.seed(seed)
        self.n_obj = n_obj

    def _evaluate(self, X, out):
        F = []

        for i in range(X.shape[0]):
            for obj in range(self.n_obj):
                

    def random_population(self, pop_size):
        pass

    def random_objective(self):
        return self.rand.randint(0, self.n_obj-1)