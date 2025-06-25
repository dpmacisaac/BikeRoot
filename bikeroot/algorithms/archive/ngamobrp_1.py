import networkx as nx
import RoutId.search.routing_algs as ra
import random
import RoutId.utils as ut

class nga_mobrp:
    def __init__(self, G, ni=10, gens=50, seed=0):
        self.rand = random.Random()
        self.rand.seed(seed)
        self.ni = ni
        self.gens = gens
        self.mutation_rate = 0.1
        self.G = BaseExceptionGroup

    def nga_mobrp(self, criteria_list, start, goal):
        self.criteria_list = criteria_list
        self.start = start
        self.goal = goal
        init_pop = self.generate_init_pop()
        for t in range(self.gens):
            pass
            
    
    def _generate_init_pop(self):
        init_pop = []
        criteria_1 = self.rand.randint(0,len(self.criteria_list))
        pop_0 = self.a_star_brp(self.start, self.goal, self.criteria_list[criteria_1])
        init_pop.append(pop_0)
        while len(init_pop) < self.ni:
            pop_1 = self.perturb(pop_0)
            init_pop.append(pop_1) # TODO: may need to check for duplicate solutions
            pop_0 = pop_1
        return init_pop
    
    def _a_star_brp(self, start, goal, criteria):
        pass

    def _perturb(self, pop, func=None):
        if func is None:
            func = self.rand(0,2)
        path_len = len(pop)
        rand_criteria = self.criteria_list[self.rand.randint(0,len(self.criteria_list))]
        n_index = self.rand.randint(0, path_len-1)
        if func == 0:
            return self.a_star_brp(pop[n_index],self.goal, rand_criteria)
        elif func == 1:
            return self.a_star_brp(self.start, pop[n_index], rand_criteria)
        else:
            n_index_2 = self.rand.randint(0, path_len-1)
            while n_index_2 == n_index:
                n_index_2 = self.rand.randint(0, path_len-1)
            if n_index_2 < n_index:
                n_index, n_index_2 = n_index_2, n_index
            return self.a_star_brp(pop[n_index], pop[n_index_2], rand_criteria)

    def _crossover(self, pop_0, pop_1):
        same_nodes = ut.get_matching_indices(pop_0, pop_1)
        if len(same_nodes) == 0:
            return pop_0, pop_1
        n_index_0, n_index_1 = self.rand.choice(same_nodes)

        return pop_0[:n_index_0] + pop_1[n_index_1:], pop_1[:n_index_1] + pop_0[n_index_0:]

        