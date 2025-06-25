import heapq
from collections import defaultdict
import numpy as np
from pymoo.util.dominator import Dominator # For pymoo dominance checks

# --- Helper Classes and Functions ---

class Label:
    _id_counter = 0

    def __init__(self, node_id, costs, predecessor_label=None):
        self.id = Label._id_counter
        Label._id_counter += 1
        self.node_id = node_id
        self.costs = tuple(float(c) for c in costs)  # k-dimensional cost vector, ensured to be tuple of floats
        self.predecessor_label = predecessor_label

    def getNode(self):
        return self.node_id

    def getCost(self):
        return self.costs

    def getPredecessorLabel(self):
        return self.predecessor_label

    def __lt__(self, other):
        if not isinstance(other, Label):
            return NotImplemented
        # Default priority: lexicographical costs. Adjust if heuristic needs specific f-values.
        return self.costs < other.costs

    def __eq__(self, other): # For set operations in Bag
        if not isinstance(other, Label):
            return False
        return self.node_id == other.node_id and self.costs == other.costs

    def __hash__(self): # For set operations in Bag
        return hash((self.node_id, self.costs))

    def reconstruct_path_nodes(self):
        path_nodes = []
        current = self
        while current is not None:
            path_nodes.append(current.node_id)
            current = current.predecessor_label
        return path_nodes[::-1]

class HeuristicHandler:
    def terminationCondition(self, current_label, destination_node_id, Q_is_empty):
        if current_label.getNode() == destination_node_id:
            pass # Potentially more complex conditions here
        return False

    def skipEdge(self, next_label, destination_node_id):
        return False

    def checkDominance(self, next_label_costs_np, bag_for_v_node):
        """
        Checks if next_label (represented by its numpy costs) is dominated by any label in bag_for_v_node.
        Returns True if next_label is NON-DOMINATED, False otherwise.
        """
        for existing_label in bag_for_v_node:
            existing_costs_np = np.array(existing_label.getCost())
            # If existing_label dominates next_label, then next_label is dominated.
            if Dominator.get_relation(existing_costs_np, next_label_costs_np) == 1:
                return False  # next_label is dominated
        return True # next_label is not dominated by any existing label in the bag for v

class HRNCIRLabelSet:
    def __init__(self, G, objective_meta, heuristic_handler=None):
        self.G = G
        self.objective_meta = objective_meta # list of dicts for cost extraction
        self.k_objectives = len(objective_meta)
        self.Bag = defaultdict(set)
        self.Q = []
        self.heuristic = heuristic_handler if heuristic_handler else HeuristicHandler()
        Label._id_counter = 0

    def _calculate_label_set(self, origin_node_id, destination_node_id):
        self.Q = []
        self.Bag = defaultdict(set)
        Label._id_counter = 0

        initial_costs = tuple([0.0] * self.k_objectives)
        L_origin = Label(node_id=origin_node_id, costs=initial_costs, predecessor_label=None)

        heapq.heappush(self.Q, L_origin)
        self.Bag[origin_node_id].add(L_origin)

        while self.Q:
            current_label = heapq.heappop(self.Q)

            # If current_label was pruned from Bag while in Q, skip it
            if current_label not in self.Bag[current_label.getNode()]:
                continue

            u_node_id = current_label.getNode()
            u_costs = current_label.getCost() # This is already a tuple of floats

            if self.heuristic.terminationCondition(current_label, destination_node_id, not self.Q):
                break

            if u_node_id not in self.G: continue

            for _, v_node_id, edge_data in self.G.edges(u_node_id, data=True):
                if not edge_data: continue # Skip if no edge data

                v_node_data = self.G.nodes[v_node_id]
                
                c_uv = []
                for i in range(self.k_objectives):
                    meta = self.objective_meta[i]
                    cost_val = 0.0
                    
                    # Edge cost component
                    if meta['edge_key'] and meta['edge_key'] in edge_data:
                        cost_val += float(edge_data[meta['edge_key']])
                    
                    # Target node penalty component (for arriving at v)
                    if meta['node_penalty_key'] and meta['node_penalty_key'] in v_node_data:
                        cost_val += float(v_node_data[meta['node_penalty_key']])
                    c_uv.append(cost_val)
                
                edge_cost_vector = tuple(c_uv)
                next_costs = tuple(u_c + e_c for u_c, e_c in zip(u_costs, edge_cost_vector))
                
                next_label = Label(node_id=v_node_id, costs=next_costs, predecessor_label=current_label)

                if self.heuristic.skipEdge(next_label, destination_node_id):
                    continue

                bag_for_v = self.Bag[v_node_id]
                next_costs_np = np.array(next_costs)

                if self.heuristic.checkDominance(next_costs_np, bag_for_v):
                    labels_in_bag_v_to_remove = []
                    for l_exist in bag_for_v:
                        # If next_label dominates l_exist
                        if Dominator.get_relation(next_costs_np, np.array(l_exist.getCost())) == 1:
                            labels_in_bag_v_to_remove.append(l_exist)
                    
                    for l_rem in labels_in_bag_v_to_remove:
                        bag_for_v.remove(l_rem)
                    
                    bag_for_v.add(next_label)
                    heapq.heappush(self.Q, next_label)
        
        return self.Bag[destination_node_id]

    def get_pareto_paths(self, origin_node_id, destination_node_id):
        """Public method to run the algorithm and return paths and their costs."""
        if not self.objective_meta:
            raise ValueError("Objective meta information not set during initialization.")
            
        pareto_labels_at_dest = self._calculate_label_set(origin_node_id, destination_node_id)
        
        results = []
        for label in pareto_labels_at_dest:
            results.append({
                "costs": label.getCost(),
                "path_nodes": label.reconstruct_path_nodes(),
                "objective_names": [meta['name'] for meta in self.objective_meta]
            })
        return results

# Example Usage (conceptual):
# if __name__ == '__main__':
#     import networkx as nx
#     G = nx.DiGraph()
#     # Add nodes with penalties
#     G.add_node('o', **{'time penalty': 0, 'safety penalty': 0, 'comfort penalty': 0})
#     G.add_node('a', **{'time penalty': 1, 'safety penalty': 2, 'comfort penalty': 0.5})
#     G.add_node('b', **{'time penalty': 0.5, 'safety penalty': 1, 'comfort penalty': 1})
#     G.add_node('d', **{'time penalty': 0, 'safety penalty': 0, 'comfort penalty': 0})

#     # Add edges with costs
#     G.add_edge('o', 'a', **{'travel distance': 10, 'travel time': 5, 'safety': 3, 'comfort': 4})
#     G.add_edge('o', 'b', **{'travel distance': 12, 'travel time': 3, 'safety': 5, 'comfort': 2})
#     G.add_edge('a', 'd', **{'travel distance': 5, 'travel time': 2, 'safety': 1, 'comfort': 3})
#     G.add_edge('b', 'd', **{'travel distance': 6, 'travel time': 4, 'safety': 2, 'comfort': 2})
#     G.add_edge('a', 'b', **{'travel distance': 3, 'travel time': 1, 'safety': 1, 'comfort': 1})

#     objective_meta_config = [
#         {'name': 'distance', 'edge_key': 'travel distance', 'node_penalty_key': None},
#         {'name': 'time',     'edge_key': 'travel time',     'node_penalty_key': 'time penalty'},
#         {'name': 'safety',   'edge_key': 'safety',          'node_penalty_key': 'safety penalty'},
#         {'name': 'comfort',  'edge_key': 'comfort',         'node_penalty_key': 'comfort penalty'}
#     ]
#     origin = 'o'
#     destination = 'd'

#     mls_solver = HRNCIRLabelSet(G, objective_meta_config)
#     pareto_solutions = mls_solver.get_pareto_paths(origin, destination)

#     print(f"Pareto solutions for path from {origin} to {destination}:")
#     for i, sol in enumerate(pareto_solutions):
#         print(f"Solution {i+1}:")
#         print(f"  Path: {sol['path_nodes']}")
#         cost_str = ", ".join([f"{name}: {cost:.2f}" for name, cost in zip(sol['objective_names'], sol['costs'])])
#         print(f"  Costs: {cost_str}")
#         # Expected output would depend on the exact graph and cost definitions
#         # Example (illustrative, not calculated for the above graph):
#         # Solution 1:
#         # Path: ['o', 'a', 'd']
#         # Costs: distance: 15.00, time: 8.00, safety: 6.00, comfort: 7.50
#         # Solution 2:
#         # Path: ['o', 'b', 'd']
#         # Costs: distance: 18.00, time: 7.50, safety: 8.00, comfort: 4.00 