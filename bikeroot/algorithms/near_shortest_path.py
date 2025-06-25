import heapq
from collections import deque
import networkx as nx
#from label import Label as l, BiLabelSet as b2
try:
    from .bilabelset import BiLabelSet, AllLabels
except ImportError:
    from bikeroot.algorithms.bilabelset import BiLabelSet, AllLabels
import json

def solve(G: nx.DiGraph,
          source: int,
          target: int,
          weight_attrs: list[str],
          **kwargs) -> list[tuple[list[int], list[float]]]:
    pass

if __name__ == "__main__":
    # Example usage
    G = nx.MultiDiGraph()
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
'''
Algorithm Notes:
weighting factors:
- lam1 = z2(xlex12) - z2(xlex21)
- lam2 = z1(xlex21) - z1(xlex12)
lex(1,2) and lex(2,1) found in initialization

upperbounds:
- used to restrict enumeration
- for every candidate xhat
- z1(xhat) <= z2(xlex21) and z2(xhat) <= z1(xlex12)
- zN = (z1(xlex21), z2(xlex12))
- minus 1 to each if its an integer problem. I am not sure in this case it is. 
- delta = max path weight (weighted sum val of improved nadir point)
- d1 and d2 keep track of the current value of the two objectives and thus allow for comparison with the respective upper bounds
'''