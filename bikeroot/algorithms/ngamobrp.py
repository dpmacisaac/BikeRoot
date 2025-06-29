from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.core.selection import Selection
from pymoo.core.duplicate     import ElementwiseDuplicateElimination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import math
import networkx as nx
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

from pymoo.core.evaluator import Evaluator
import osmnx as ox
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.population import Population
import random
import numpy as np
from bikeroot import BikeNetwork, CostApplicator, Cost, NgaMobrp_Cost, alg_utils
from bikeroot import RouteDuplicateElim

def solve(G: nx.MultiDiGraph,
          src,
          dst,
          criteria: list[str],debug=False,
          **kwargs) -> list[tuple[list[int], list[float]]]:
    seed = kwargs.get('seed', 0)

    alg = kwargs.get('alg','dynamic')
    survival = kwargs.get('surv', 'ngamobrp')
    options = {}

    problem = NGAMOBRPProblem(G,src, dst ,seed=seed, criteria=criteria)
    population_size = kwargs.get('population_size', 100)
    mutation_prob = kwargs.get('mut_prob', 0.05)
    if survival == 'ngamobrp':
        options['survival'] = NGAMOBRPSurvival(population_size)
    if alg == 'dynamic':
        algorithm = DynamicNSGA2(
            pop_size   = population_size,
            sampling   = NGAMOBRPSampling(population_size),
            crossover  = NGAMOBRPCrossover(),
            mutation   = NGAMOBRPMutation(prob=mutation_prob),
            selection  = NGAMOBRPSelection(),
            eliminate_duplicates = RouteDuplicateElim(problem),
            **options
        )
    else:
        algorithm = NSGA2(
            pop_size   = population_size,
            sampling   = NGAMOBRPSampling(population_size),
            crossover  = NGAMOBRPCrossover(),
            mutation   = NGAMOBRPMutation(),
            selection  = NGAMOBRPSelection(),
            eliminate_duplicates = RouteDuplicateElim(problem),
            **options
        )

    n_gen = kwargs.get('n_gen', 200)
    termination = get_termination("n_gen", n_gen)
    params = {}
    if debug:
        class DebugCB(Callback):
            def __init__(self):
                super().__init__()
            def notify(self, algorithm):
                gen = algorithm.n_gen
                print(f"Gen {gen:02d} | pop.F[0] = {algorithm.pop.get('F')[0]}")
        
        params['callback'] = DebugCB()  # drop if you don’t want console spam

    res= minimize(
        problem,                  # your NGAMOBRPProblem
        algorithm,
        termination,    # drop if you don’t want console spam
        seed     = seed,
        verbose  = False,           # prints generational stats
        **params
    )

    routes = []
    for idx in  range(len(res.pop.get("X"))):
        route = problem.get_route(res.pop.get("X")[idx][0])
        cost = {}
        for i, obj in enumerate(problem.obj_names):
            cost[obj] = float(res.pop.get("F")[idx][i])
        routes.append((route, cost))

    return routes

class DynamicNSGA2(NSGA2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _next(self):
        super()._next()               # NSGA-II as usual

        # keep the whole survivor set as the new fixed size
        new_size = len(self.pop)      # can grow or shrink
        if new_size != self.pop_size:
            self.pop_size           = new_size
            self.n_offsprings       = new_size
            self.survival.n_survive = new_size   # for the *next* gen

class NGAMOBRPProblem(Problem):
    def __init__(self, graph, src, dst, seed=0, criteria=None):
        if criteria is not None:
            n_obj = len(criteria)
        else:
            n_obj = 4
        super().__init__(n_var=1, n_obj=n_obj, n_constr=0,
                         xl=0.0,
                         elementwise_evaluation=False,  # evaluate whole pop
                         eliminate_duplicates=True)

        self.graph = graph
        self.route_storage = {}
        self.fitness_storage = {}
        self.route_hash = {}
        self.rand = random.Random(seed)
        self.src = src
        self.dst = dst
        self.obj_names = criteria if criteria is not None else ['travel distance', 'travel time', 'safety', 'comfort']

    def store_route(self, route):
        key = tuple(route)            # lists aren’t hashable
        if key in self.route_hash:    # seen before → re-use index
            return self.route_hash[key]

        idx = len(self.route_storage) # brand-new route
        self.route_storage[idx] = route
        self.fitness_storage[idx] = None
        self.route_hash[key] = idx
        return idx
    
    def get_route(self, idx):
        return self.route_storage[int(idx)]
    
    def print_solutions(self):
        """Print all routes in storage."""
        for idx, route in self.route_storage.items():
            print(f"Route {idx}: {route}")
            print(f"Fitness: {self.fitness_storage[idx]}")

    def remove_route(self, idx):
        """Remove route from storage."""
        idx = int(idx)
        if idx in self.route_storage:
            del self.route_storage[idx]
            del self.fitness_storage[idx]
        else:
            raise ValueError(f"Route index {idx} not found in storage.")

    def _evaluate(self, X, out):
        F = np.zeros((len(X), self.n_obj))
        for r, row in enumerate(X):
            idx = int(row[0])
            # If we have not seen this route before, compute & cache
            if self.fitness_storage[idx] is None:
                route = self.get_route(idx)
                self.fitness_storage[idx] = self._compute_objectives(route)

            F[r] = self.fitness_storage[idx]
        out["F"] = F

    def random_objective(self):
        return self.rand.randint(0, self.n_obj-1)
    
    def _compute_objectives(self, route):
        costs = [0.0 for _ in range(self.n_obj)]

        for n1, n2 in zip(route[:-1], route[1:]):
            # per-node contributions   (first node of edge)
            # ndata = self.graph.nodes[n1]
            # t     += float(ndata['time penalty'])
            # safe  += float(ndata['safety penalty'])
            # comfy += float(ndata['comfort penalty'])

            # per-edge contributions
            edata = self.graph.get_edge_data(n1, n2)[0]
            for i, obj in enumerate(self.obj_names):
                if obj in edata:
                    costs[i] += float(edata[obj])
                else:
                    raise ValueError(f"Objective '{obj}' not found in edge data.")
        return np.array(costs)
    
    def perturb(self, version, route):
        n = len(route)
        if n < 2:
            return route
        node_i = self.rand.randint(1, n-2)
        obj_idx = self.random_objective()
        perturbed = route
        if version == 1:
            perturbed = route[:node_i] + self.a_star(route[node_i], self.dst, obj_idx)
        elif version == 2:
            perturbed = self.a_star(self.src, route[node_i], obj_idx) + route[node_i+1:]
        elif version == 3:
            node_i2 = self.rand.randint(1, n-2)
            while node_i2 == node_i:
                node_i2 = self.rand.randint(1, n-2)
            if node_i2 < node_i:
                node_i, node_i2 = node_i2, node_i
            # print(node_i, node_i2)
            perturbed =  route[:node_i] + self.a_star(route[node_i],route[node_i2], obj_idx) + route[node_i2+1:]
    
        return perturbed

    @staticmethod
    def haversine_distance(cords_1, cords_2):
        R = 6_371_000                               # metres
        lat1, lon1 = cords_1
        lat2, lon2 = cords_2
        lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))
    
    @staticmethod
    def haversine_heuristic(node_1, node_2):
        """
        Calculate the haversine distance between two nodes in a graph
        :param graph: the graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the haversine distance between the two nodes in meters
            """
        # Get the coordinates of the two nodes
        cords_1 = node_1['y'], node_1['x']
        cords_2 = node_2['y'], node_2['x']

        # Calculate the haversine distance
        return NGAMOBRPProblem.haversine_distance(cords_1, cords_2)

    @staticmethod
    def time_heuristic(node_1, node_2):
        """
        Calculate the haversine distance between two nodes in a graph
        :param graph: the graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the haversine distance between the two nodes in meters
            """
        # Get the coordinates of the two nodes
        cords_1 = node_1['y'], node_1['x']
        cords_2 = node_2['y'], node_2['x']

        # Calculate the haversine distance
        return NGAMOBRPProblem.haversine_distance(cords_1, cords_2)/7.582*math.exp(0.1072*(-1))
    @staticmethod
    def safety_heuristic(node_1, node_2):
        """
        Calculate the haversine distance between two nodes in a graph
        :param graph: the graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the haversine distance between the two nodes in meters
            """
        # Get the coordinates of the two nodes
        cords_1 = node_1['y'], node_1['x']
        cords_2 = node_2['y'], node_2['x']

        # Calculate the haversine distance
        return NGAMOBRPProblem.haversine_distance(cords_1, cords_2) * 0.15
    @staticmethod
    def comfort_heuristic(node_1, node_2):
        """
        Calculate the haversine distance between two nodes in a graph
        :param graph: the graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the haversine distance between the two nodes in meters
            """
        # Get the coordinates of the two nodes
        cords_1 = node_1['y'], node_1['x']
        cords_2 = node_2['y'], node_2['x']

        # Calculate the haversine distance
        return NGAMOBRPProblem.haversine_distance(cords_1, cords_2) * 0.57

    def a_star(self, src, dst, obj_idx):
        heuristics = {
            'travel distance': lambda u, v: NGAMOBRPProblem.haversine_heuristic(self.graph.nodes[u], self.graph.nodes[v]),
            'travel time'   : lambda u, v: NGAMOBRPProblem.time_heuristic(self.graph.nodes[u], self.graph.nodes[v]),
            'safety'        : lambda u, v: NGAMOBRPProblem.safety_heuristic(self.graph.nodes[u], self.graph.nodes[v]),
            'comfort'       : lambda u, v: NGAMOBRPProblem.comfort_heuristic(self.graph.nodes[u], self.graph.nodes[v])
        }
        return alg_utils.a_star(
            self.graph, src, dst,
            heuristic=heuristics[self.obj_names[obj_idx]],
            weight=self.obj_names[obj_idx]
        )

class NGAMOBRPSampling(Sampling):
    """
        Sampling operator for the NGAMOBRP algorithm.
        Code Complete
    """

    def __init__(self, init_pop_size, **kwargs):
        # print("initializing NGAMOBRPSampling...")
        super().__init__(**kwargs)
        self.init_pop_size = init_pop_size
        # print("NGAMOBRPSampling initialized.")

    def _do(self, problem: NGAMOBRPProblem, n_samples: int, **kwargs) -> np.ndarray:
        
        """
        Returns an (n_samples, 1) array with dtype=float where each entry
        is the index into problem.route_storage.
        """
        # print("Running custom sampling for NGAMOBRP...")
        routes_idx = []

        # --- step 1: random criterion for P0 --------------------------------
        obj_idx = problem.random_objective()  # random objective

        # initialize first solution based on obj
        P0 = problem.a_star(problem.src, problem.dst, obj_idx)
        
        # append route
        routes_idx.append(problem.store_route(P0))

        # --- step 2: generate rest by perturbing ----------------------------
        while len(routes_idx) < n_samples:
            # pick one of the three perturb versions at random
            perturb_version = problem.rand.randint(1,3)
            # print(f"Perturbing with version {perturb_version} on route {P0}")
            P1 = problem.perturb(perturb_version, P0)
            # print(P1)
            idx = problem.store_route(P1)
            routes_idx.append(idx)
            P0 = P1
        # --- step 3: return as NumPy array ---------------------------------
        # print(f"Generated {len(routes_idx)} routes for NGAMOBRP sampling.")
        # print(problem.route_storage)
        arr = np.array(routes_idx).reshape(-1, 1).astype(float)
        # print(arr)
        return arr

class NGAMOBRPMutation(Mutation):
    """
    Mutation operator for the NGAMOBRP algorithm.
    *Representation* — the decision vector has a single gene: the float/int
    index into `problem.route_storage`.
    """

    def __init__(self, prob=0.05, **kwargs):
        """
        Parameters
        ----------
        prob : float
            Per-individual mutation probability (default 0.4).
        """
        # print("initializing NGAMOBRPMutation...")
        super().__init__(prob=prob, **kwargs)
        self.pr = prob
        # print("NGAMOBRPMutation initialized.")

    # ------------------------------------------------------------------ #
    def _do(self,problem:NGAMOBRPProblem, X:np.ndarray, **kwargs):
        """
        Parameters
        ----------
        problem : NGAMOBRPProblem
            Must expose:
            • route_storage  - list / array of full paths
            • store_route(P) - returns index of P (deduplicates if you wish)
            • perturb(v, P)  - your perturbation operator
            • rand           - NumPy Generator for reproducibility
        X : ndarray, shape (n_indiv, 1)
            Each entry is the route index (dtype float/int).

        Returns
        -------
        Y : ndarray, same shape & dtype as X
            Mutated population.
        """
        # print("Running custom mutation for NGAMOBRP...")
        rng = problem.rand            # short-hand
        Y = X.copy()

        for i in range(len(X)):
            z = rng.random()          # random number in [0, 1)
            if z > float(self.pr):         # skip according to prob
                continue
            idx_old = int(X[i, 0])
            P0 = problem.get_route(idx_old)
            # --- requested random mutation -------------------------------- #
            P1 = problem.perturb(rng.randint(1, 3), P0)
            idx_new = problem.store_route(P1)
            Y[i, 0] = idx_new                    # keep dtype identical
            # print(f'X: {X[i, 0]} -> {idx_old} \n P0: {P0}  \n P1: {P1} \n Y: {Y[i, 0]}')
        return Y

class NGAMOBRPSelection(Selection):
    """
    Replace NSGA-II's binary tournament.  The paper favours individuals
    from earlier fronts but *without* crowding.  Here's a stub that simply
    picks uniformly *within* the first two ranks but you should adapt it.

    code complete
    """

    def __init__(self):
        # n_parents = 2 for mating; everything returns pairs
        # print("initializing NGAMOBRPSelection...")
        super().__init__()
        self.n_parents = 2
        # print("NGAMOBRPSelection initialized.")

    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        # print("Running custom selection for NGAMOBRP...")
        F = pop.get("F")
        # print(f'Selecting {n_select} pairs of parents with {n_parents} parents from {len(F)} individuals. ')
        # print(F)
        # print(pop.get("X"))
        # print(problem.route_storage)
        

        rng = problem.rand               # short-hand
        # # 1) Choose two *different* objectives at random
        obj_a, obj_b = rng.sample(range(0,problem.n_obj), 2)
        # # 2) Build two ranked lists of *indices* (smallest value first)
        list_a = np.argsort(F[:, obj_a]).tolist()
        list_b = np.argsort(F[:, obj_b]).tolist()
        # print(f'lista {list_a} \n listb {list_b}')

        pairs = []   # list‑of‑lists shaped (n_matings, 2)

        # 3) Repeatedly take the top of each list
        while len(pairs) < n_select and list_a and list_b:

            idx_a = list_a[0]
            idx_b = list_b[0]

            # 3a) If both lists point to the same individual,
            #      fall back to the runner‑up in list‑B.
            if idx_a == idx_b:
                if len(list_b) < 2:
                    break                      # not enough material left
                idx_b = list_b[1]

            pairs.append([idx_a, idx_b])

            # 4) Remove both chosen indices from *both* lists so
            #    neither can appear in later matings this round.
            for L in (list_a, list_b):
                if idx_a in L:
                    L.remove(idx_a)
                if idx_b in L:
                    L.remove(idx_b)
        if pairs == []:
            if len(F) == 1:
                # if only one solution, return it as a pair
                # print("Only one solution available, returning it as a pair.")
                # return self pairing of only solution
                pairs = [[list_a[0], list_a[0]] for _ in range(n_select)]
            if len(F) == 2:
                # if only two solutions, return them as pairs
                # print("Only two solutions available, returning them as pairs.")
                # return self pairing of only two solutions
                pairs = [[list_a[0], list_a[1]] for _ in range(n_select)]
        # 5) Return as a NumPy array – shape must be (n_matings, 2)
        arr = np.asarray(pairs, dtype=np.int64)
        # print(arr)
        return arr

class NGAMOBRPCrossover(Crossover):
    """
    Crossover operator for the NGAMOBRP algorithm.
    """

    def __init__(self, **kwargs):
        # print("initializing NGAMOBRPCrossover...")
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)
        # print("NGAMOBRPCrossover initialized.")

    def _do(self, problem, pop, **kwargs):
        """
        input:
            pop = np.ndarray of shape (n_parents, n_matings, n_var)

        output:
            offspring = np.ndarray of shape (n_offspring, n_matings, n_var)

        """
        # print("Running custom crossover for NGAMOBRP...")
                # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = pop.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(pop, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):

            # get the first and the second parent
            if len(pop) ==  1:
                Y[0, k, 0] = pop[0, k, 0]
                continue

            a, b = pop[0, k, 0], pop[1, k, 0]
            if a == b:
                # If both parents are the same, just copy the parent
                Y[0, k, 0] = a
                Y[1, k, 0] = b
                continue

            route_a = problem.get_route(a)
            route_b = problem.get_route(b)
            common_nodes = list(set(route_a) & set(route_b))
            common_nodes.remove(problem.src)
            common_nodes.remove(problem.dst)
            if len(common_nodes) == 0:
                # No common nodes, just copy the parents
                Y[0, k, 0], Y[1, k, 0] = a, b
                continue

            # Randomly select a common node to be the crossover point
            crossover_point = problem.rand.choice(common_nodes)

            # Find the index of the crossover point in both routes
            idx_a = route_a.index(crossover_point)
            idx_b = route_b.index(crossover_point)

            # Create the offspring route
            child_a = route_a[:idx_a + 1] + route_b[idx_b + 1:]
            child_b = route_b[:idx_b + 1] + route_a[idx_a + 1:]

            new_a = problem.store_route(child_a)
            new_b = problem.store_route(child_b)
            # join the character list and set the output
            Y[0, k, 0], Y[1, k, 0] = new_a, new_b

        return Y

class NGAMOBRPSurvival(Survival):
    def __init__(self, pop_size, filter_infeasible=True):
        super().__init__(filter_infeasible=filter_infeasible)
        self.pop_size = pop_size
        self.persisent_popsize = pop_size
        
    def _do(self, problem, pop, n_survive, *args, **kwargs):
        F = pop.get("F")
        fronts = NonDominatedSorting().do(F)
        survivors = []
        f = 0
        while len(survivors) < self.persisent_popsize and f < len(fronts):
            survivors.extend(fronts[f].tolist())
            f += 1
        # print('survivors: ', len(survivors), 'of', len(pop), 'from', f, 'fronts')
        return pop[np.asarray(survivors, dtype=int)]  
