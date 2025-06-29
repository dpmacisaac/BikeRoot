import numpy as np
from bikeroot import alg_utils
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def calculate_refpoint(G, src, dst, criteria):
    criteria_paths = {c: None for c in criteria}
    criteria_costs = {c: [] for c in criteria}
    for c in criteria:
        path = alg_utils.a_star(G, src, dst, weight=c)
        criteria_paths[c] = path if path else None
        for k in criteria:
            cost_k = alg_utils.total_path_cost(G, path, weight=k) if path else -1
            criteria_costs[k].append(cost_k)
    print(f'criteria_costs: {criteria_costs}')
    criteria_refpoints = {c: -1 for c in criteria}
    for c in criteria:
        criteria_refpoints[c] = max(criteria_costs[c]) if criteria_costs[c] else -1
    return criteria_refpoints

def paths_costs_split(paths_costs):
    paths, costs = zip(*paths_costs)
    return list(paths), list(costs)

def remove_dominated_solutions(criteria, paths, costs):
    """Removes dominated solutions based on the criteria. Returns only non-dominated ones."""
    if not costs:
        return paths, costs

    # Convert cost dicts to a 2D array for dominance checking
    F = np.array([[c[k] for k in criteria] for c in costs], dtype=float)

    # Find indices of non-dominated solutions
    nd_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)

    # Filter and return only the non-dominated paths and costs
    nd_paths = [paths[i] for i in nd_indices]
    nd_costs = [costs[i] for i in nd_indices]
    return nd_paths, nd_costs