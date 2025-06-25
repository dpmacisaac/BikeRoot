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
    problem = NGAMOBRPProblem(G,src, dst ,seed=seed, criteria=criteria)
    population_size = kwargs.get('population_size', 100)
    algorithm = NSGA2(
        pop_size   = population_size,
        sampling   = NGAMOBRPSampling(population_size),
        crossover  = NGAMOBRPCrossover(),
        mutation   = NGAMOBRPMutation(),
        selection  = NGAMOBRPSelection(),
        eliminate_duplicates = RouteDuplicateElim(problem)
    )

    n_gen = kwargs.get('n_gen', 200)
    termination = get_termination("n_gen", n_gen)
    if debug:
        class DebugCB(Callback):
            def __init__(self):
                super().__init__()
            def notify(self, algorithm):
                gen = algorithm.n_gen
                print(f"Gen {gen:02d} | pop.F[0] = {algorithm.pop.get('F')[0]}")
        res= minimize(
            problem,                  # your NGAMOBRPProblem
            algorithm,
            termination,
            callback = DebugCB(),     # drop if you don’t want console spam
            seed     = seed,
            verbose  = True           # prints generational stats
        )
    else:
        res= minimize(
            problem,                  # your NGAMOBRPProblem
            algorithm,
            termination,    # drop if you don’t want console spam
            seed     = seed,
            verbose  = False           # prints generational stats
        )
    routes = []
    for idx in  range(len(res.pop.get("X"))):
        route = problem.get_route(res.pop.get("X")[idx][0])
        cost = {}
        for i, obj in enumerate(problem.obj_names):
            cost[obj] = float(res.pop.get("F")[idx][i])
        routes.append((route, cost))

    return routes

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
        self.rand = random.Random(seed)
        self.src = src
        self.dst = dst
        self.obj_names = criteria if criteria is not None else ['travel distance', 'travel time', 'safety', 'comfort']

    def store_route(self, route):
        """Save route, return its index so operators can build chromosomes."""
        idx = len(self.route_storage)
        self.route_storage[idx] = route
        self.fitness_storage[idx] = None
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
        objective = self.random_objective()
        #print(str(version) + "  " +str(node_i) + "  " + str(self.obj_names[objective]))
        if version == 1:
            return route[:node_i] + self.a_star(route[node_i], self.dst, objective)
        elif version == 2:
            return self.a_star(self.src, route[node_i], objective) + route[node_i+1:]
        elif version == 3:
            node_i2 = self.rand.randint(1, n-2)
            while node_i2 == node_i:
                node_i2 = self.rand.randint(1, n-2)
            if node_i2 < node_i:
                node_i, node_i2 = node_i2, node_i
            # print(node_i, node_i2)
            new =  route[:node_i] + self.a_star(route[node_i],route[node_i2], objective) + route[node_i2+1:]
            # print(new)
            return new
        return route
    @staticmethod
    def haversine_distance(cords_1, cords_2):
        """
        Calculate the haversine distance between two sets of coordinates
        :param cords_1: the first set of coordinates
        :param cords_2: the second set of coordinates
        :return: the haversine distance between the two sets of coordinates in meter
        """
        # Radius of the Earth in kilometers
        R = 6371

        # Convert latitude and longitude from degrees to radians
        lat1, lon1 = cords_1
        lat2, lon2 = cords_2
        lat1, lon1, lat2, lon2 = map(lambda x: x * (3.1415 / 180), [lat1, lon1, lat2, lon2])

        # Calculate the haversine distance
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * (dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c * 1000
    
    @staticmethod
    def haversine_heuristic(graph, node_1, node_2):
        """
        Calculate the haversine distance between two nodes in a graph
        :param graph: the graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the haversine distance between the two nodes in meters
            """
        # Get the coordinates of the two nodes
        cords_1 = graph.nodes[node_1]['y'], graph.nodes[node_1]['x']
        cords_2 = graph.nodes[node_2]['y'], graph.nodes[node_2]['x']

        # Calculate the haversine distance
        return NGAMOBRPProblem.haversine_distance(cords_1, cords_2)

    @staticmethod
    def time_heuristic(graph, node_1, node_2):
        """
        Calculate the haversine distance between two nodes in a graph
        :param graph: the graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the haversine distance between the two nodes in meters
            """
        # Get the coordinates of the two nodes
        cords_1 = graph.nodes[node_1]['y'], graph.nodes[node_1]['x']
        cords_2 = graph.nodes[node_2]['y'], graph.nodes[node_2]['x']

        # Calculate the haversine distance
        return NGAMOBRPProblem.haversine_distance(cords_1, cords_2)
    @staticmethod
    def safety_heuristic(graph, node_1, node_2):
        """
        Calculate the haversine distance between two nodes in a graph
        :param graph: the graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the haversine distance between the two nodes in meters
            """
        # Get the coordinates of the two nodes
        cords_1 = graph.nodes[node_1]['y'], graph.nodes[node_1]['x']
        cords_2 = graph.nodes[node_2]['y'], graph.nodes[node_2]['x']

        # Calculate the haversine distance
        return NGAMOBRPProblem.haversine_distance(cords_1, cords_2)
    @staticmethod
    def comfort_heuristic(graph, node_1, node_2):
        """
        Calculate the haversine distance between two nodes in a graph
        :param graph: the graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the haversine distance between the two nodes in meters
            """
        # Get the coordinates of the two nodes
        cords_1 = graph.nodes[node_1]['y'], graph.nodes[node_1]['x']
        cords_2 = graph.nodes[node_2]['y'], graph.nodes[node_2]['x']

        # Calculate the haversine distance
        return NGAMOBRPProblem.haversine_distance(cords_1, cords_2)

    def a_star(self, src, dest, objective):
        """
        A* algorithm to find the shortest path from src to dest.
        """
        # Use the graph and the cost function to find the path
        heuristic = None
        if objective == 0:
            heuristic = NGAMOBRPProblem.haversine_heuristic
        elif objective == 1:
            heuristic = NGAMOBRPProblem.time_heuristic
        elif objective == 2:
            heuristic = NGAMOBRPProblem.safety_heuristic
        elif objective == 3:
            heuristic = NGAMOBRPProblem.comfort_heuristic
        path = alg_utils.a_star(self.graph, src, dest, heuristic=heuristic, weight=self.obj_names[objective])
        return path

class NGAMOBRPSampling(Sampling):
    """
        Sampling operator for the NGAMOBRP algorithm.
        Code Complete
    """

    def __init__(self, init_pop_size, **kwargs):
        super().__init__(**kwargs)
        self.init_pop_size = init_pop_size

    def _do(self, problem: NGAMOBRPProblem, n_samples: int, **kwargs) -> np.ndarray:
        
        """
        Returns an (n_samples, 1) array with dtype=float where each entry
        is the index into problem.route_storage.
        """
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
            P1 = problem.perturb(perturb_version, P0)
            # print(P1)
            idx = problem.store_route(P1)
            routes_idx.append(idx)
            P0 = P1
        # --- step 3: return as NumPy array ---------------------------------
        return np.array(routes_idx).reshape(-1, 1).astype(float)

class NGAMOBRPMutation(Mutation):
    """
    Mutation operator for the NGAMOBRP algorithm.
    *Representation* — the decision vector has a single gene: the float/int
    index into `problem.route_storage`.
    """

    def __init__(self, prob=0.4, **kwargs):
        """
        Parameters
        ----------
        prob : float
            Per-individual mutation probability (default 0.4).
        """
        super().__init__(prob=prob, **kwargs)
        self.pr = prob

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
        super().__init__()
        self.n_parents = 2

    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        X = pop.get("X")
        F = pop.get("F")
        # F = pop.get("F")                     # (N, 4) objective matrix
        rng = problem.rand               # short-hand
        # # 1) Choose two *different* objectives at random
        obj_a, obj_b = rng.sample(range(0,problem.n_obj), 2)
        # # 2) Build two ranked lists of *indices* (smallest value first)
        list_a = np.argsort(F[:, obj_a]).tolist()
        list_b = np.argsort(F[:, obj_b]).tolist()

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

        # 5) Return as a NumPy array – shape must be (n_matings, 2)
        return np.asarray(pairs, dtype=np.int64)

class NGAMOBRPCrossover(Crossover):
    """
    Crossover operator for the NGAMOBRP algorithm.
    """

    def __init__(self, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)

    def _do(self, problem, pop, **kwargs):
        """
        input:
            pop = np.ndarray of shape (n_parents, n_matings, n_var)

        output:
            offspring = np.ndarray of shape (n_offspring, n_matings, n_var)

        """
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
    """
    Implements NGA rule:
        • Do nondominated sorting.
        • Keep fronts F1, F2, … until the first that *exceeds* pop_size,
          *but still keep that whole front* (population can grow).
        • No crowding distance pruning.
        • Also drop exact duplicates (optional).

    We attach rank info to pop for the selector.
    """

    def __init__(self, pop_size, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size

    # ------------- pymoo API ------------
    def _do(self, problem, pop, n_survive, *args, **kwargs):
        F = pop.get("F")      # (N, n_obj)
        nds = NonDominatedSorting()  # ndarray int
        ranks = nds.do(F)
        pop.set("rank", ranks)

        survivors = []
        current_front = 0
        while True:
            idx = np.where(ranks == current_front)[0]
            if len(idx) == 0:
                break
            survivors.extend(idx.tolist())
            if len(survivors) >= self.pop_size:
                # keep WHOLE current front even if we overflow
                current_front += 1
                continue
            current_front += 1

        return survivors[:self.pop_size]  # return only the first pop_size individuals
