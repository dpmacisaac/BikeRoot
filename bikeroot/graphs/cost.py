import math
import networkx as nx
import numpy as np
import osmnx as ox

from . import utils
from .bikenetwork import BikeNetwork

class Cost:
    def __init__(self):
        # criteria:
        pass

    @staticmethod
    def get_edge_cost(self, edge, graph):
        raise NotImplementedError("get_edge_cost must be implemented by subclass")

    @staticmethod
    def get_node_cost(self, node, graph):
        raise NotImplementedError("get_node_cost must be implemented by subclass")
    
    @staticmethod
    def has_node_data(self):
        raise NotImplementedError("has_node_data must be implemented by subclass")

class CostApplicator:

    @staticmethod
    def apply_cost_functions(bike_net:BikeNetwork, cost:Cost):
        G = bike_net.graph
        G_new = nx.MultiDiGraph()
        
        # Process Nodes
        for node in G.nodes(data=True):
            attrs = cost.get_node_cost(node, G)
            G_new.add_node(node[0], **attrs)

        # Process Edges
        for edge in G.edges(data=True):
            attrs = cost.get_edge_cost(edge,G)
            G_new.add_edge(edge[0], edge[1], **attrs)

        # for node in G_new.nodes(data=True):
        #     print(node)
        # for edge in G_new.edges(data=True):
        #     print(edge)
        if cost.has_node_data():
            # If the cost function has node data, we need to split the vertices
            # G_new = utils.vertex_split_attr(G_new, cost.criteria_labels)
            utils.push_node_costs_to_in_edges(G_new, cost.criteria_labels)

        return G_new

def main():
    pass

if __name__ == "__main__":
    main()