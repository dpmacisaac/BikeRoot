from collections import deque, defaultdict
import networkx as nx
import math, logging, json, heapq
from bikeroot import BikeNetwork, alg_utils
import numpy as np
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def solve(G: nx.MultiDiGraph,
          src,
          dst,
          criteria: list[str],
          H=None,
          **kwargs) -> list[tuple[list[int], list[float]]]:
    debug = kwargs.get('debug', False)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.CRITICAL)

    nm = NamoaData(criteria)

    if H == 'tc':
        H = tc_heuristic(G, dst, criteria)
    else:
        dst_coords = (G.nodes[dst]['y'], G.nodes[dst]['x'])
        H = haversine_heurisitic(G, dst_coords, criteria)

    F_src = H(src)

    logging.info(f"Starting NAMOA* from {src} to {dst} with criteria {criteria} and heur {F_src}")

    src_label_id = nm.new_label(src, [0.0 for crit in criteria],F_src, None)
    nm.open_add_label_id(src_label_id)

    while True:
        # 2 Check Termination
        if nm.open_empty():
            logging.info("Open list is empty")
            return nm.find_final_paths()
        
        # 3 Path Selection
        label_n = nm.open_get_alt()
        n, g_n, F_n, prev_n_id = nm.get_label_info(label_n)

        logging.info(f"Selected node {n} with cost {g_n} and F set {F_n}")

        nm.move_to_closed(label_n, n)

        # 4 Solution Recording
        if n == dst:
            logging.info(f"Found a path to destination {dst} with cost {g_n}")
            # add gn to costs
            nm.add_to_goal_costs(label_n, g_n)
            # for all alts in OPEN, eliminate all vectors dominated by gn
            # elimated from open all alternatives such that Fx is empty
            continue

        # 5 Path Expansion
        for m in G.successors(n):
            if not nm.does_produce_cycle(label_n, m):
                # (a) calculate cost of new path
                g_m = [g_n[idx] + G[n][m][0][crit] for idx,crit in enumerate(criteria)]

                if nm.is_new_node(m):
                    # calculate F_m, filter dominated by costs
                    F_m = calculate_F_set(m, g_m, criteria, H)

                    # if Fm not emtpy, put (m, gm, Fm) in open list, add m gm to Gop, label gm to n
                    if nm.filter_by_costs(F_m):
                        m_label_id = nm.new_label(m, g_m, F_m, label_n)
                        nm.open_add_label_id(m_label_id)
                    
                else:
                    if nm.cost_relation_equals(m, g_m):
                        # label with g_m a pointer to n
                        m_label_id = nm.new_label(m, g_m, F_m, label_n)
                        nm.open_add_label_id(m_label_id)
                    elif nm.cost_relation_nondominated(m, g_m):
                        dominates = nm.find_dominated_labels_at_node(g_m, m)
                        nm.remove_dominated_labels(dominates)
                        F_m = calculate_F_set(m, g_m, criteria, H)

                        # if Fm not emtpy, put (m, gm, Fm) in open list, add m gm to Gop, label gm to n
                        if nm.filter_by_costs(F_m):
                            m_label_id = nm.new_label(m, g_m, F_m, label_n)
                            nm.open_add_label_id(m_label_id)
                        
                        # interesting path foiund:
                        # eliminate labels in Gop and Gcl that are dominated by g_m
                        # eliminate corresponding eliminated vectors from open

                        # calculate F_m, filter dominated by costs
                        # if Fm not emtpy, put (m, gm, Fm) in open list, add m gm to Gop, label gm to n

def tc_heuristic(graph, dst, criteria):
    G_rev = graph.reverse(copy=True)
    tc_heur = {node: [math.inf for _ in criteria] for node in graph.nodes}
    for i, crit in enumerate(criteria):
        try:
            lengths = nx.single_source_dijkstra_path_length(G_rev, dst, weight=crit)
            for node, cost in lengths.items():
                tc_heur[node][i] = cost
        except nx.NetworkXNoPath:
            pass  # unreachable nodes stay as is
    def tc_heurisitic_fn(node):
        """
        Calculate the scalarized heuristic between two nodes in a graph
        :param node: the node to calculate the heuristic for
        :return: the scalarized heuristic for the node
        """
        # Example heuristic: total cost to destination
        return tc_heur[node]
    return tc_heurisitic_fn

def calculate_F_set(node, g_cost, criteria, heuristic):
    H_n = heuristic(node)
    return [g_cost[idx] + H_n[idx] for idx in range(len(criteria))]

def haversine_heurisitic(graph, dst_coords, criteria):
    def haversine_heurisitic_fn(node):
        """
        Calculate the scalarized heuristic between two nodes in a graph
        :param node_1: the first node
        :param node_2: the second node
        :return: the scalarized heuristic between the two nodes
        """
        distance = haversine_distance((node['y'], node['x']), dst_coords)

        mods = {'travel distance': 1,
                'travel time': 1/7.582*math.exp(0.1072*(-0.92)),
                'safety': 0.15,
                'comfort': 0.57}

        return [distance * mods[k] for k in criteria]
    return lambda u: haversine_heurisitic_fn(graph.nodes[u])

def haversine_distance(cords_1, cords_2):
    R = 6_371_000                               # metres
    lat1, lon1 = cords_1
    lat2, lon2 = cords_2
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))



class NamoaData:
    """Holds the mutable state structures used by NAMOA*."""

    def __init__(self, criteria: list[str]):
        self.criteria = criteria

        # id to label, unsorted list of (node, g_cost, F_set, prev_label_id)
        self._label_store = {}

        # [label_ids]
        self.open_list: list[tuple] = [] 

        # {node: [ids]}
        self.gop: defaultdict = defaultdict(list)  # open labels
        self.gcl: defaultdict = defaultdict(list)  # closed labels

        # Current Pareto front of goal costs
        self.goal_costs: list[tuple] = []
        self.goal_labels: list[int] = []

        # Predecessor pointers for path reconstruction
        self.preds: dict = {}

    def new_label(self, node, g_cost, F_set, prev_label_id):
        """
        Create a new label for a node with its associated cost and F set.
        Returns the label ID.
        """
        label_id = len(self._label_store)
        self._label_store[label_id] = (node, g_cost, F_set, prev_label_id)
        return label_id
    
    def get_label_info(self, label_id):
        """
        Retrieve the information associated with a label ID.
        Returns (node, g_cost, F_set, prev_label_id).
        """
        if label_id in self._label_store:
            return self._label_store[label_id]
        else:
            raise ValueError(f"Label ID {label_id} not found in label store.")

    def open_add_label_id(self, label_id):
        self.open_list.append(label_id)
        self.gop_add_label_id(label_id)

    def open_empty(self) -> bool:
        """Check if the open list is empty."""
        return len(self.open_list) == 0

    def open_get_alt(self):
        logging.debug(f"Open list before selection: {self.open_list}")
        for label_id_a in self.open_list:
            dominated = False
            _, _, F_set_a, _ = self._label_store[label_id_a]
            logging.debug(f"Checking label {label_id_a} in open list. with F_set {F_set_a} ")
            for label_id_b in self.open_list:
                if label_id_a == label_id_b:
                    continue
                _, _, F_set_b, _ = self._label_store[label_id_b]

                if self.dominated(F_set_a,F_set_b):
                    dominated = True
                    break

            if not dominated:
                logging.info(f"Selected label {label_id_a}")
                return label_id_a

        logging.warning("No alternative found in open list.")
        return None
    
    def gop_add_label_id(self, label_id):
        """Add a label ID to the open list."""
        node, _, _, prev_label_id = self._label_store[label_id]
        if label_id not in self.gop[node]:
            self.gop[node].append(label_id)
        self.preds[label_id] = prev_label_id
    
    def move_to_closed(self, label_id, node):
        """Move a node from the open list to the closed list."""
        if label_id in self.open_list:
            self.open_list.remove(label_id)
        if label_id in self.gop[node]:
            self.gop[node].remove(label_id)
        if label_id not in self.gcl[node]:
            self.gcl[node].append(label_id)
            logging.info(f"Moved label {label_id} for node {node} to closed list.")
        else:
            logging.warning(f"Label {label_id} not found in open list for node {node}.")
    
    def filter_by_costs(self, F_set):
        for goal_cost in self.goal_costs:
            if self.dominates(goal_cost, F_set):
                logging.debug(f'Label {goal_cost} dominated cost {F_set}, removing from open list.')
                return False
        logging.info(f'Fset {F_set} not dominated by costs')
        return True

    def does_produce_cycle(self, label_n_id, m):
        """Check if expanding to node m produces a cycle."""
        curr_label_id = label_n_id

        while True:
            logging.debug(f"Checking for cycles: current label {curr_label_id}, node {m}")
            # 1) Compare the node stored in this ancestor label
            if self._label_store[curr_label_id][0] == m:
                logging.info(f"Cycle detected: label {label_n_id} repeats node {m}.")
                return True   # node repeats → cycle

            # 2) Climb to the next ancestor, if any
            if curr_label_id not in self.preds:
                break         # reached the root (start label)

            curr_label_id = self.preds[curr_label_id]

            if curr_label_id is None:
                break
        logging.info(f"No cycle detected for label {label_n_id} and node {m}.")
        return False
    
    def is_new_node(self, m):
        return (m not in self.gop or len(self.gop[m]) == 0) and (m not in self.gcl or len(self.gcl[m]) == 0)

    def cost_relation_equals(self, m, g_m):
        for label_id in self.gop[m]:
            _, g_m_b,_, _ = self._label_store[label_id]
            if all(g_m[i] == g_m_b[i] for i in range(len(self.criteria))):
                logging.info(f"Label {label_id} is equal to {g_m}, not adding to open list.")
                return True
        for label_id in self.gcl[m]:
            _, g_m_b,_, _ = self._label_store[label_id]
            if all(g_m[i] == g_m_b[i] for i in range(len(self.criteria))):
                logging.info(f"Label {label_id} is equal to {g_m}, not adding to open list.")
                return True
        logging.info(f"No equal label found for {g_m} in Gop or Gcl.")
        return False

    def dominates(self, a, b):
        # check if a dominates b
        return all(a[i] <= b[i] for i in range(len(self.criteria))) and any(a[i] < b[i] for i in range(len(self.criteria)))

    def dominated(self, a, b):
        # check if a is dominated by b
        return all(a[i] >= b[i] for i in range(len(self.criteria))) and any(a[i] > b[i] for i in range(len(self.criteria)))


    def cost_relation_nondominated(self, m, g_m):
        for label_id in self.gop[m]:
            _, g_m_b,_, _ = self._label_store[label_id]
            if self.dominated(g_m,g_m_b):
                return False
        for label_id in self.gcl[m]:
            _, g_m_b,_, _ = self._label_store[label_id]
            if self.dominated(g_m,g_m_b):
                return False
        logging.info(f"No dominated label found for {g_m} in Gop or Gcl.")
        return True

    def add_to_goal_costs(self, label_id, g_n):
        survivors  = [g for g in self.goal_costs if not self.dominates(g_n, g)]
        if any(self.dominates(g, g_n) for g in survivors):
            return                         # new goal is dominated – ignore it
        self.goal_costs = survivors + [g_n]
        self.goal_labels.append(label_id)
        dominated = self.find_dominated_labels_in_open(g_n)
        self.remove_dominated_labels(dominated)

    def remove_dominated_labels(self, dominated):
        for label_id_b in dominated:
            node_b,_,_, _ = self._label_store[label_id_b]
            if label_id_b in self.open_list:
                self.open_list.remove(label_id_b)
                logging.info(f"Removed dominated label {label_id_b} from open list.")
            if label_id_b in self.gop[node_b]:
                self.gop[node_b].remove(label_id_b)
                logging.info(f"Removed dominated label {label_id_b} from Gop.")

    def find_dominated_labels_in_open(self,g_m):
        dominated = []
        for label_id_b in self.open_list:
            if g_m == label_id_b:
                continue
            _, _, F_set_b, _ = self._label_store[label_id_b]
            if self.dominates(g_m, F_set_b):
                dominated.append(label_id_b)
                logging.info(f"Label {label_id_b} is dominated by {g_m}, removing from open list.")
        return dominated
    
    def find_dominated_labels_at_node(self,g_m, m):
        dominated = []
        labels = self.gop[m] + self.gcl[m]
        for label_id_b in labels:
            if g_m == label_id_b:
                continue
            _, g_m_b,_ , _ = self._label_store[label_id_b]

            if self.dominates(g_m, g_m_b):
                dominated.append(label_id_b)
                logging.info(f"Label {label_id_b} is dominated by {g_m}, removing from open list.")
        return dominated
    
    def find_final_paths(self):
        paths_costs = []
        for goal_id in self.goal_labels:
            path = []
            node, cost, _, prev_label_id = self._label_store[goal_id]
            path.append(node)
            while prev_label_id is not None:
                node, _, _, prev_label_id = self._label_store[prev_label_id]
                path.append(node)
            path.reverse()
            paths_costs.append((path, {k: cost[i] for i, k in enumerate(self.criteria)}))
        return paths_costs


if __name__ == "__main__":
    location = "Spokane, WA"
    loc_name = location.split(",")[0].lower().replace(" ", "_")
    criteria = ['travel distance', 'travel time', 'safety', 'comfort']
    G = BikeNetwork(location)
    G.load_graph(f'data/small_maps/{loc_name}_1000_withcost.graphml')
    G.weights_to_floats(criteria)
    print(f"Loaded graph for {location} with {len(G.graph.nodes)} nodes and {len(G.graph.edges)} edges.")
    src= '48925897'
    dst= '50968572'
    paths = solve(G.graph, src=src, dst=dst, criteria=criteria)