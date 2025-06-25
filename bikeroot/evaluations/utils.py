import numpy as np
from bikeroot import alg_utils

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