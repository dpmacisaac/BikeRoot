
import networkx as nx
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
import numpy as np
from bikeroot import NGAMOBRPCrossover, NGAMOBRPMutation, NGAMOBRPSampling, NGAMOBRPSelection
from bikeroot import NGAMOBRPProblem

def solve(G: nx.MultiDiGraph,
          src,
          dst,
          criteria: list[str],debug=False,
          **kwargs) -> list[tuple[list[int], list[float]]]:
    seed = kwargs.get('seed', 0)
    problem = NGAMOBRPProblem(G,src, dst ,seed=seed, criteria=criteria)
    population_size = kwargs.get('population_size', 100)
    ref_dirs = get_reference_directions("uniform", len(criteria), n_partitions=7)
    population_size = len(ref_dirs) if population_size < len(ref_dirs) else population_size
    algorithm = MOEAD(
        sampling   = NGAMOBRPSampling(population_size),
        crossover  = NGAMOBRPCrossover(),
        mutation   = NGAMOBRPMutation(),
        selection  = NGAMOBRPSelection(),
        ref_dirs=ref_dirs
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