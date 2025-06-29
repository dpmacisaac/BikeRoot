from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import math
import heapq
import numpy as np
import time

def remove_duplicate_paths(paths):
    pass

def haversine_distance(cords_1, cords_2):
    R = 6_371_000                               # metres
    lat1, lon1 = cords_1
    lat2, lon2 = cords_2
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def haversine_heuristic(graph, node_1, node_2):
    """
    Calculate the haversine distance between two nodes in a graph
    :param graph: the graph
    :param node_1: the first node
    :param node_2: the second node
    :return: the haversine distance between the two nodes in meters
    """
    # Get the coordinates of the two nodes
    cords_1 = graph.nodes[node_1]['y'], graph.nodes[node_1]['x']
    cords_2 = graph.nodes[node_2]['y'], graph.nodes[node_2]['x']

    # Calculate the haversine distance
    return haversine_distance(cords_1, cords_2)

def total_path_cost(G, path, criteria):
    """Compute the total cost of a path based on a specific edge attribute.
    
    Args:
        G (nx.Graph): The graph.
        path (list): A list of node IDs representing the path.
        weight (str): The edge attribute to sum.
    
    Returns:
        float: Total cost of the path.
    """
    if isinstance(criteria, str):
        return sum(float(G.get_edge_data(u, v)[0][criteria]) for u, v in zip(path[:-1], path[1:]))

    return {crit: sum(float(G.get_edge_data(u, v)[0][crit]) for u, v in zip(path[:-1], path[1:])) for crit in criteria}

def a_star(G, src, dst, heuristic=haversine_heuristic, weight="length"):
    """
    A* algorithm to find the shortest path between two nodes in a graph
    :param graph: the graph to search
    :param start: the starting node
    :param goal: the goal node
    :return: the shortest path between start and goal (Iterable[list[int]])
    """
    return nx.astar_path(G, src, dst, heuristic=heuristic, weight=weight)
    # open_set = set([src]) # TODO: change to priority queue
    # closed_set = set()
    # g = {src: 0}
    # #h = {start: heuristic(graph, start, goal)}
    # f = {src: g[src] + heuristic(graph, src, dst)}
    # parent = {src: None}
    
    # while(open_set):
    #     # Get the node in the open set with the lowest f value
    #     current = min(open_set, key=lambda x: f[x])

    #     # Reached Goal
    #     if current == dst:
    #         path = []
    #         while current != src:
    #             path.append(current)
    #             current = parent[current]
    #         path.append(src)
    #         path.reverse()
    #         return path
        
    #     open_set.remove(current)
    #     closed_set.add(current)

    #     successors = graph.out_edges(current)
    #     for successor_edge in successors:
    #         successor = successor_edge[1]

    #         if successor in closed_set:
    #             continue
    #         x = float(graph.get_edge_data(current, successor)[0][weight])
    #         succ_g = g[current] + x # dist from start to curr + curr to succ
    #         succ_h = heuristic(graph, successor, dst)
    #         succ_f = succ_g + succ_h

    #         if successor not in open_set:
    #             open_set.add(successor)
    #         elif succ_g >= g[successor]:
    #             continue
            
    #         # this is best path so far
    #         parent[successor] = current
    #         g[successor] = succ_g
    #         f[successor] = succ_f

    # return None

def a_star_heap(graph, start, goal, heuristic=haversine_heuristic, weight="length"):
    """
    A* algorithm to find the shortest path between two nodes in a graph
    :param graph: the graph to search
    :param start: the starting node
    :param goal: the goal node
    :return: the shortest path between start and goal (Iterable[list[int]])
    """
    open_set = []
    closed_set = set()
    g = {start: 0}
    #h = {start: heuristic(graph, start, goal)}
    f = {start: g[start] + heuristic(graph, start, goal)}
    parent = {start: None}
    heapq.heappush(open_set, (f[start], start))

    while(open_set):
        # Get the node in the open set with the lowest f value
        current = heapq.heappop(open_set)[1]

        # Reached Goal
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path
        
        closed_set.add(current)

        successors = graph.out_edges(current)
        for successor_edge in successors:
            successor = successor_edge[1]

            if successor in closed_set:
                continue

            succ_g = g[current] + graph.get_edge_data(current, successor)[0][weight] # dist from start to curr + curr to succ
            succ_h = heuristic(graph, successor, goal)
            succ_f = succ_g + succ_h

            if successor not in [i[1] for i in open_set]:
                heapq.heappush(open_set, (succ_f, successor))
            elif succ_g >= g[successor]:
                continue
            
            # this is best path so far
            parent[successor] = current
            g[successor] = succ_g
            f[successor] = succ_f

    return None

if __name__=='__main__':
    G = ox.io.load_graphml(filepath="data/spokane.graphml")
    print('graph loaded')
    point_1 = ox.geocoder.geocode("1101 S Westcliff Pl, Spokane, WA")

    node_1 = ox.distance.nearest_nodes(G, point_1[1], point_1[0])
    print(f'node_1: {node_1}')
    point_2 = ox.geocoder.geocode("601 E Riverside Ave, Spokane, WA 99202")

    node_2 = ox.distance.nearest_nodes(G, point_2[1], point_2[0])
    print(f'node_2: {node_2}')

    start = time.time()
    path = a_star(graph=G,src=node_1,dst=node_2,heuristic=haversine_heuristic, weight='length')
    end = time.time()
    print('Time taken a_star:', end-start)

    print(total_path_cost(G, path, weight='length'))

    start = time.time()
    path = a_star_heap(graph=G,start=node_1,goal=node_2,heuristic=haversine_heuristic, weight='length')
    end = time.time()
    print('Time taken a_star_heap:', end-start)
    print(total_path_cost(G, path, weight='length'))

    ox_path = ox.shortest_path(G, node_1, node_2, weight='length')
    print(total_path_cost(G, ox_path, weight='length'))

    if ox_path==path:
        print('Success')