import heapq
from collections import deque
import networkx as nx
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
    modNodes = deque()
    modNodes.append(src)
    source_cost = {w: 0.0 for w in criteria}

    labels = BiLabelSet(criteria)
    labels.add_label(node_at=src, cost=source_cost)
    # ind = 0
    while len(modNodes) > 0:
        # if ind % 100 == 0:
        #     print(f'Processed {ind} nodes, {len(modNodes)} in queue')
        # ind += 1
        i = modNodes.popleft()
        # print(f'processing node {i}')
        for j in G.successors(i):
            cost_ij = G.get_edge_data(i, j)[0]
            for k in criteria:
                cost_ij[k] = float(cost_ij[k])
            # print(f'processing edge {i} -> {j} with cost {cost_ij}')
            changed = labels.node_merge(i, j, cost_ij)
            if changed and j not in modNodes:
                modNodes.append(j)
    # labels.print_by_nodes(target)
    return labels.backtrack(src, dst)

if __name__ == "__main__":
    # Example usage
    G = nx.DiGraph()
    G.add_edge(0, 1, time=1, dist= 5)
    G.add_edge(0, 2, time=5, dist=5)
    G.add_edge(1, 2, time=2, dist=1)
    G.add_edge(0, 3, time=5, dist=4)
    G.add_edge(3, 2, time=0, dist=0)

    source = 0
    target = 2
    weight_attrs = ['time', 'dist']

    result = solve(G, source, target, weight_attrs)
    print(result)  # Should print the paths and their costs