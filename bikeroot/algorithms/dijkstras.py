import heapq
import networkx as nx
from typing import Any, Dict, List, Tuple, Union, Optional


def dijkstra(G:nx.MultiDiGraph, source, target= None, weight='weight'):
    '''
    inputs:
        G: a MultiDiGraph
        source: the source node
        target: the target node (optional). If none, the shortest path to all nodes is computed.
        weight: the edge attribute to use as weight (default is 'weight')
    '''
    if target is None:
        return nx.single_source_dijkstra(G, source, weight=weight)
    else:
        return nx.dijkstra_path(G, source, target, weight=weight)




def lex_dijkstra(G: nx.MultiDiGraph, criteria, source, target= None):
    



if __name__ == "__main__":
    # Example usage
    G = nx.MultiDiGraph()
    attrs = {'time':1, 'dist':5}
    G.add_edge(0, 1,**attrs)
    attrs = {'time':5, 'dist':5}
    G.add_edge(0, 2, **attrs)

    attrs = {'time':2, 'dist':1}
    G.add_edge(1, 2, **attrs)

    attrs = {'time':5, 'dist':4}
    G.add_edge(0, 3, **attrs)
    attrs = {'time':0, 'dist':0}
    G.add_edge(3, 2, **attrs)

    source = 0
    target = 2
    weight_attrs = ['time', 'dist']


    print(lex_dijkstra(G, source, None, criteria=weight_attrs))
