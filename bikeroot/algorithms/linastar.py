import networkx as nx
import math, itertools, random, logging
from math import comb
from bikeroot import alg_utils
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def solve(G: nx.MultiDiGraph,
          src,
          dst,
          criteria: list[str],debug=False,
          **kwargs) -> list[tuple[list[int], list[float]]]:
    log = kwargs.get('log_level', 'critical')
    if log == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif log == 'critical':
        logging.getLogger().setLevel(logging.CRITICAL)
    crit_count = len(criteria)
    sol_count = kwargs.get('sol_count', crit_count*4)
    
    # print(simplex_lattice(sol_count, crit_count))
    prefs = preference_weights(sol_count, crit_count)
    pref = {k:-1 for k in criteria}
    paths = []
    for idx in range(sol_count):
        pref = {k: float(prefs[idx][i]) for i, k, in enumerate(criteria)}
        path = alg_utils.a_star(G, src, dst, scalarized_heuristic(pref,G), scalarized_cost(pref))
        if path not in paths:
            paths.append(path)
    paths_costs = []
    for path in paths:
        cost = alg_utils.total_path_cost(G, path, criteria)
        paths_costs.append((path, cost))
    return paths_costs


def simplex_lattice(sol_count: int, crit_count: int):
    # ---- step 1: smallest H with enough points
    H = 0
    while comb(H + crit_count - 1, crit_count - 1) < sol_count:
        H += 1

    # ---- step 2: weak compositions of H into m parts
    def weak_compositions(H, m):
        # place m-1 dividers into H+m-1 slots
        for dividers in itertools.combinations(range(H + m - 1), m - 1):
            parts = []
            last = -1
            for d in dividers + (H + m - 1,):
                parts.append(d - last - 1)
                last = d
            yield parts

    lattice = [tuple(h / H for h in comp)
               for comp in weak_compositions(H, crit_count)]

    # ---- step 3: optionally shorten and reorder
    # (a) move the pure criteria to the front
    axis = [tuple(1.0 if i == k else 0.0 for i in range(crit_count))
            for k in range(crit_count)]
    interior = [w for w in lattice if w not in axis]
    lattice = axis + interior

    # (b) trim to exactly sol_count if necessary
    return lattice[:sol_count]

def lattice_points(m: int, H: int):
    """Das & Dennis simplex lattice spacing 1/H."""
    def rec(remaining, k, prefix):
        if k == m - 1:
            yield prefix + [remaining]
        else:
            for h in range(remaining + 1):
                yield from rec(remaining - h, k + 1, prefix + [h])
    for ints in rec(H, 0, []):
        yield np.array(ints, dtype=float) / H

def build_candidate_pool(m: int, sol_count: int):
    """Smallest H s.t. lattice size ≥ sol_count."""
    H = 1
    while comb(H + m - 1, m - 1) < sol_count:
        H += 1
    pool = []
    for h in range(1, H + 1):
        pool.extend(lattice_points(m, h))
    # remove duplicates that appear for different H
    uniq = []
    for p in pool:
        if not any(np.allclose(p, q) for q in uniq):
            uniq.append(p)
    return uniq

def preference_weights(sol_count: int, crit_count: int, seed=42):
    random.seed(seed)
    m = crit_count
    pool = build_candidate_pool(m, sol_count)

    # step 1  – extremes
    selected = [np.eye(m)[i] for i in range(m)]

    # step 2  – centroid
    centroid = np.full(m, 1.0 / m)
    selected.append(centroid)

    # remove the ones already chosen from pool
    pool = [p for p in pool
            if not any(np.allclose(p, s) for s in selected)]

    # step 3  – greedy maximin
    while len(selected) < sol_count:
        # compute nearest distance to the current set
        dists = [min(np.linalg.norm(p - s) for s in selected) for p in pool]
        if not dists: break
        max_d = max(dists)
        # collect the *front* (all ties)
        front = [p for p, d in zip(pool, dists) if abs(d - max_d) < 1e-12]
        random.shuffle(front)              # random choice inside the front
        for p in front:
            selected.append(p)
            if len(selected) == sol_count:
                break
        pool = [p for p in pool
                if not any(np.allclose(p, s) for s in selected)]

    return selected[:sol_count]


def scalarized_cost(prefs):
    def scalarized_cost_fn(u, v, attr):
        attr = attr[0] 
        return sum(prefs[k] * attr[k] for k in prefs)
    return scalarized_cost_fn

def scalarized_heuristic(prefs, graph):
    def scalarized_heuristic_fn(node_1, node_2):
        """
        Calculate the scalarized heuristic between two nodes in a graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the scalarized heuristic between the two nodes
        """
        distance = haversine_distance((node_1['y'], node_1['x']), (node_2['y'], node_2['x']))
        mods = {'travel distance': 1,
                'travel time': 1/7.582*math.exp(0.1072*(-1)),
                'safety': 0.15,
                'comfort': 0.57}
        return sum(prefs[k] * distance * mods[k] for k in prefs)
    return lambda u, v: scalarized_heuristic_fn(graph.nodes[u], graph.nodes[v])

def haversine_distance(cords_1, cords_2):
    R = 6_371_000                               # metres
    lat1, lon1 = cords_1
    lat2, lon2 = cords_2
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))
