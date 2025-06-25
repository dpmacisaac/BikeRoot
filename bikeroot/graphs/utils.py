from geopy.distance import geodesic
import math
import networkx as nx
from typing import Iterable, Union, Dict, Any

def haversine_distance(cords_1, cords_2):
    """
    Calculate the haversine distance between two sets of coordinates
    :param cords_1: the first set of coordinates
    :param cords_2: the second set of coordinates
    :return: the haversine distance between the two sets of coordinates in meter
    """
    # Radius of the Earth in kilometers
    R = 6371

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = cords_1
    lat2, lon2 = cords_2
    lat1, lon1, lat2, lon2 = map(lambda x: x * (3.1415 / 180), [lat1, lon1, lat2, lon2])

    # Calculate the haversine distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * (dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c * 1000

def get_edge_coords(G, edge):
    coords_u = (G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y'])
    coords_v = (G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y'])
    return coords_u, coords_v

def get_bounding_box_from_point_or_place(center_point, dist_meters=1000):
    """
    Get north/south/east/west points and a bounding box around a city or lat/lon point.

    Args:
        center_point:(lat, lon) tuple
        dist_meters: distance in meters to extend from center in each cardinal direction

    Returns:
        dict with:
            - center: (lat, lon)
            - north, south, east, west: each as (lat, lon)
            - bounding_box: (north, south, east, west)
    """
    lat, lon = center_point

    # Compute points in each direction
    north = geodesic(meters=dist_meters).destination((lat, lon), 0)
    south = geodesic(meters=dist_meters).destination((lat, lon), 180)
    east = geodesic(meters=dist_meters).destination((lat, lon), 90)
    west = geodesic(meters=dist_meters).destination((lat, lon), 270)

    bounding_box = (
        west.longitude,
        south.latitude,
        east.longitude,
        north.latitude
    )
    # west, south, east, north
    return {
        "center": center_point,
        "north": (north.latitude, north.longitude),
        "south": (south.latitude, south.longitude),
        "east": (east.latitude, east.longitude),
        "west": (west.latitude, west.longitude),
        "bounding_box": bounding_box  # (west, south, east, north)/(left, bottom, right, top)
    }

def vertex_split_attr(
    G: nx.DiGraph,
    criteria_labels: list = None,
    fill_missing: bool = True,
    default_val: float = 0
) -> nx.DiGraph:
    # Union of all node-attribute keys (time, dist, …)
    crit_set = set(criteria_labels)          # faster membership tests
    H = nx.MultiDiGraph()

    # 1. split every vertex -----------------------------------------------
    for v, ndata in G.nodes(data=True):
        # separate node attrs into "keep" vs "move"
        node_attrs: Dict[str, Any] = {k: v for k, v in ndata.items()
                                      if k not in crit_set}
        edge_attrs: Dict[str, Any] = {k: v for k, v in ndata.items()
                                      if k in crit_set}

        # fill any missing criteria keys on the internal edge
        if fill_missing:
            for k in crit_set:
                edge_attrs.setdefault(k, default_val)

        vin, vout = f"{v}_in", f"{v}_out"
        H.add_node(vin, **node_attrs)
        H.add_node(vout, **node_attrs)
        H.add_edge(vin, vout, **edge_attrs)

    # 2. redirect each original edge --------------------------------------
    def add_edge(src: str, dst: str, data: Dict[str, Any]):
        attrs = data.copy()
        if fill_missing:
            for k in crit_set:
                attrs.setdefault(k, default_val)
        H.add_edge(src, dst, **attrs)

    for u, v, edata in G.edges(data=True):
        add_edge(f"{u}_out", f"{v}_in", edata)
        if not G.is_directed():              # keep symmetry for undirected
            add_edge(f"{v}_out", f"{u}_in", edata)

    return H


def push_node_costs_to_in_edges(G, criteria: list[str]):
    """
    Move each node cost in `criteria` onto *all* incoming edges.
    Modifies the graph in place.
    """
    for v, ndata in G.nodes(data=True):
        for k in criteria:
            c = ndata.get(k, 0)
            if c == 0:
                continue
            for u, _, edata in G.in_edges(v, data=True):
                edata[k] = edata.get(k, 0) + c
            ndata[k] = 0         # or `del ndata[k]` if you prefer