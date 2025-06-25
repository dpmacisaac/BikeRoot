import heapq
from collections import deque
import networkx as nx
#from label import Label as l, BiLabelSet as b2
try:
    from .bilabelset import BiLabelSet, AllLabels
except ImportError:
    from bikeroot.algorithms.bilabelset import BiLabelSet, AllLabels
import json

def solve(G: nx.MultiDiGraph,
          src,
          dst,
          criteria: list[str],
          **kwargs) -> list[tuple[list[int], list[float]]]:
    source_cost = {w: 0.0 for w in criteria}
    labels = BiLabelSet(criteria)

    id = labels.add_label(node_at=src, cost=source_cost)
    allLabels = AllLabels(criteria, labels)
    allLabels.add_label(id=id)

    while not allLabels.empty():
        id = allLabels.pop()
        if id is None:
            break
        label = labels.get_label(id)
        node_at = label['node_at']
        cost = label['cost']
        for j in G.successors(node_at):
            edge_cost = G.get_edge_data(node_at, j)[0]
            # edge_cost = G[node_at][j]
            new_cost = {k: cost[k] + float(edge_cost[k]) for k in criteria}
            added, deleted = labels.label_merge(node_at, new_cost, id, j)
            for new_id in added:
                allLabels.add_label(id=new_id)
            for del_id in deleted:
                allLabels.remove_label(del_id)
    return labels.backtrack(src, dst)

if __name__ == "__main__":
    # Example usage
    G = nx.DiGraph()
    G.add_edge(0, 1, time=1, dist= 5)
    G.add_edge(0, 2, time=5, dist=5)
    G.add_edge(1, 2, time=2, dist=1)
    G.add_edge(0, 3, time=5, dist=4)
    G.add_edge(3, 2, time=0, dist=0)
    #G.add_edge(1, 2, time=10, dist=14)

    source = 0
    target = 2
    weight_attrs = ['time', 'dist']

    result = solve(G, source, target, weight_attrs)
    print(result)  # Should print the paths and their costs