import heapq
from collections import deque
import networkx as nx
import json
import math

try:
    from .multilabelset import MultiLabelSet
except ImportError:
    from bikeroot.algorithms.multilabelset import MultiLabelSet
from bikeroot import alg_utils, BikeNetwork

def solve(G: nx.MultiDiGraph,
          src,
          dst,
          criteria: list[str],heuristic_prune=None, heuristic_skip=None, heuristic_checkdom=None,
          **kwargs) -> list[tuple[list[int], list[float]]]:
    
    heuristic_prune=Pruning(dst= dst, G=G, **heuristic_prune)
    heuristic_skip = Skip(G= G, src=src, dst=dst,**heuristic_skip)
    heuristic_checkdom = CheckDominance(G=G, criteria=criteria, **heuristic_checkdom)

    multilabelset = MultiLabelSet(criteria)
    lo_id = multilabelset.add_label(node_at=src, cost={w: 0.0 for w in criteria}, prev_id=None)
    multilabelset.add_to_queue(lo_id)

    while multilabelset.queue_empty() is False:
        curr_id = multilabelset.queue_pop()
        curr_label = multilabelset.get_label(curr_id)
        curr_node = curr_label['node_at']
        curr_cost = curr_label['cost']
        # print(f"Current Node: {curr_node}, Cost: {curr_cost}, ID: {curr_id}")
        if heuristic_prune is not None:
            if heuristic_prune.terminationCondition(curr_label) is True:
                # print(f"Termination condition met at node {curr_node} with cost {curr_cost}.")
                break
        for v in G.successors(curr_node):
            if v is None:
                continue
            edge_data = G.get_edge_data(curr_node, v)[0]
            new_cost = {k: curr_cost[k] + float(edge_data[k]) for k in criteria}
            new_edge = {'node_at': v, 'cost': new_cost, 'prev_id': curr_id}

            if heuristic_skip is not None and heuristic_skip.skip(multilabelset, new_edge):
                continue
            if heuristic_checkdom is not None:
                ifAdd, toRemove = heuristic_checkdom.checkdom(new_edge, multilabelset)
                if ifAdd:
                    new_id = multilabelset.add_label(node_at=v, cost=new_cost, prev_id=curr_id)
                    multilabelset.add_to_queue(new_id)
                    if toRemove:
                        for remove_id in toRemove:
                            multilabelset.remove_from_queue(remove_id)
                            multilabelset.remove_label(remove_id)



    return multilabelset.backtrack(src,dst)



class Skip:
    def __init__(self, type, G, **kwargs):
        self.type = type
        self.G = G
        if type == 'ellipse':
            src = kwargs.get('src')
            dst = kwargs.get('dst')
            k = float(kwargs.get('ab', 1.0))
            dmin = float(kwargs.get('dmin', 0.0))
            nodea_data = G.nodes[src]
            
            self.src_coords = (nodea_data['y'], nodea_data['x'])
            nodeb_data = G.nodes[dst]
            self.dst_coords = (nodeb_data['y'], nodeb_data['x'])
            od = alg_utils.haversine_distance(self.src_coords, self.dst_coords)
            c = 0.5 * od
            # 1. aspect-ratio rule
            a = c / math.sqrt(1.0 - 1.0 / (k * k))

            # 2. minimum over-hang rule
            if a - c < dmin:
                a = c + dmin
            self.bounds = 2 * a
           

    def skip(self, multilabelset, new_edge):
        if self.type == 'mls':
            return self.mls_skip(multilabelset, new_edge)
        elif self.type == 'ellipse':
            return self.ellipse_skip(new_edge=new_edge)
        else:
            raise ValueError(f"Unknown skip type: {self.type}")

    def mls_skip(self, multilabelset, new_edge):
        prev_node = multilabelset.get_label(new_edge['prev_id'])['node_at']
        if prev_node == new_edge['node_at']:
            return True
        return False

    def ellipse_skip(self, new_edge):
        v = new_edge['node_at']
        v_coords = (self.G.nodes[v]['y'], self.G.nodes[v]['x'])
        ov = alg_utils.haversine_distance(self.src_coords, v_coords)
        vd = alg_utils.haversine_distance(self.dst_coords, v_coords)
        # print(f"ov: {ov}, vd: {vd}, bounds: {self.bounds}")
        if abs(ov) + abs(vd) > self.bounds:
            return True
        return False
class CheckDominance:
    def __init__(self, type, G, criteria, **kwargs):
        self.G = G
        self.criteria = criteria
        self.type = type
        if type == 'epsilon':
            self.epsilon = float(kwargs.get('epsilon', 0.1))

    def checkdom(self, new_edge, multilabelset):
        if self.type == 'epsilon':
            return self.epsilon_check(new_edge, multilabelset)
        else:
            raise ValueError(f"Unknown dominance check type: {self.type}")

    def epsilon_check(self, new_edge, multilabelset):
        epsilon_cost = {attr: new_edge['cost'][attr] * (1+ self.epsilon) for attr in self.criteria}

        if multilabelset.get_labels_by_node(new_edge['node_at']) is []:
            return True, []

        for label_id in multilabelset.get_labels_by_node(new_edge['node_at']):
            l_cost = multilabelset.get_label(label_id)['cost']
            if all(l_cost[attr] < epsilon_cost[attr] for attr in self.criteria):
                return False, []
        remove = []   
        for label_id in multilabelset.get_labels_by_node(new_edge['node_at']):
            l_cost = multilabelset.get_label(label_id)['cost']
            l_epsilon_cost = {attr: l_cost[attr] * (1 + self.epsilon) for attr in self.criteria}
            new_cost = new_edge['cost']
            if all(new_cost[attr] < l_epsilon_cost[attr] for attr in self.criteria):
                remove.append(label_id)
        return True, remove

class Pruning:
    def __init__(self, type, dst, G, **kwargs):
        self.dst = dst
        self.G = G
        self.type = type
        if type == 'ratio':
            self.reached_destination = False
            self.criteria = kwargs.get('criteria')
            self.alpha = float(kwargs.get('alpha', 1.0))


    def terminationCondition(self, label):
        if self.type == 'ratio':
            return self.ratio_termination(label)

    def ratio_termination(self, label):
        cost = label['cost']
        u = label['node_at']
        self.uppercost = math.inf
        if u == self.dst and not self.reached_destination:
            self.uppercost = self.alpha * cost[self.criteria]
            self.reached_destination = True
        if cost[self.criteria] > self.uppercost:
            return True
        return False

if __name__ == "__main__":

    bike_net = BikeNetwork("Spokane, WA")
    bike_net.load_graph(f'data/spokane_500_cost_vertex_split.graphml')

    src = '11498515455'
    dst = '12077918977'
    criteria = ['travel distance', 'travel time','safety', 'comfort']
    skip = Skip(type= 'ellipse', G=bike_net.graph, source=src, target=dst, ab=1.25)
    #skip = Skip(type='mls', G=bike_net.graph)
    checkdom = CheckDominance(type='epsilon', G=bike_net.graph, criteria=criteria, epsilon=0.05)

    prune = Pruning(type='ratio', dst=dst, G=bike_net.graph, criteria='travel time', alpha=1.6)

    result = solve(bike_net.graph, src, dst, criteria, heuristic_skip=skip, heuristic_checkdom=checkdom, heuristic_prune=prune)
    print(result)  # Should print the paths and their costs
    print(f"Number of paths found: {len(result)}")