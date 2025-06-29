import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
from geopy.distance import geodesic
import elevation as elev
import os
import shutil
import numpy as np

from . import utils

class BikeNetwork():
    edge_tags = {
            "surface",                 # ground, gravel, asphalt…
            "bicycle",                 # yes / designated
            "junction",                # roundabout
            "bridge",                  # yes / movable / suspension
            "crossing",                # zebra / marked / island…
            "traffic_calming",         # bump / table
            # highway, oneway, length… are already in the default list
    }
    node_tags = {
            # highway=* node features
            "highway",                 # traffic_signals / stop / give_way / elevator / steps
            "crossing",                # zebra / uncontrolled / traffic_signals
            "traffic_calming",         # bump / table  (bumps are often nodes)
    }

    def __init__(self, location):
        ox.settings.useful_tags_way = tuple(
            sorted(set(ox.settings.useful_tags_way).union(BikeNetwork.edge_tags))
        )
        ox.settings.useful_tags_node = tuple(
            sorted(set(ox.settings.useful_tags_node).union(BikeNetwork.node_tags))
        )

        if isinstance(location, str):
            center_point = ox.geocoder.geocode(location)
        else:
            center_point = location  # assume (lat, lon) tuple
        self.center_point = center_point
        self.location = location
        self.location_name = location.split(",")[0].lower().replace(" ", "_")
    
    def download_data(self, dist_meters=1000, elevation=False, save_file=None, network_type='all', simplify=True, retain_all=False, truncate_by_edge=False, custom_filter=None):
        if os.path.exists(save_file):
            print(f"Graph already exists at {save_file}. Loading from file.")
            return self.load_graph(save_file)

        points = utils.get_bounding_box_from_point_or_place(self.center_point, dist_meters=dist_meters)

        self.graph = ox.graph_from_bbox(points["bounding_box"], network_type="all", simplify=simplify, retain_all=retain_all, truncate_by_edge=truncate_by_edge, custom_filter=custom_filter)


        if elevation:
            if save_file is not None:
                ox.save_graphml(self.graph, filepath=f'{save_file.replace(".graphml", "")}_noelevation.graphml')
            if not os.path.exists(f'{os.getcwd()}/.cache/SRTM1/{self.location_name}_{dist_meters}.tif'):
                elev.clip(bounds=points["bounding_box"], output=f'{self.location_name}_{dist_meters}.tif', cache_dir=f'{os.getcwd()}/.cache')
                # Add elevation data to the graph
            
            self.graph = ox.add_node_elevations_raster(self.graph, filepath=f'{os.getcwd()}/.cache/SRTM1/{self.location_name}_{dist_meters}.tif', band=1)
            self.graph = ox.add_edge_grades(self.graph)
            os.remove(f'{save_file.replace(".graphml", "")}_noelevation.graphml')

        if "crs" not in self.graph.graph:
            self.graph.graph["crs"] = "EPSG:4326"
        if save_file is not None:
            ox.save_graphml(self.graph, filepath=save_file)

        return self.graph
    
    def update_graph(self, graph):
        self.graph = graph

    def save_graph(self, filepath):
        ox.save_graphml(self.graph, filepath)

    def load_graph(self, filepath):
        self.graph = ox.load_graphml(filepath, node_dtypes={"osmid": str})
        if "crs" not in self.graph.graph:
            self.graph.graph["crs"] = "EPSG:4326"
        return self.graph
    
    def weights_to_floats(self, criteria):
        """
        Convert edge weights to floats based on the specified criteria.
        """
        if not isinstance(criteria, list):
            criteria = [criteria]
        
        for edge in self.graph.edges(data=True):
            for criterion in criteria:
                if criterion in edge[2]:
                    edge[2][criterion] = float(edge[2][criterion])
                else:
                    raise ValueError(f"Criterion '{criterion}' not found in edge data.")
        
        return self.graph


    def prints(self, part, limit=None):
        if part == "nodes":
            print(f"Number of nodes: {self.graph.number_of_nodes()}")
            nodes = self.graph.nodes(data=True)
            print("\nNodes:")
            i = 0
            for node in nodes:
                if limit is not None and i >= limit:
                    break
                print(node)
                i += 1
        elif part == "edges":
            print(f"Number of edges: {self.graph.number_of_edges()}")
            edges = self.graph.edges(data=True)
            print("\nEdges:")
            i = 0
            for edge in edges:
                if limit is not None and i >= limit:
                    break
                print(edge)
                i += 1
        else:
            raise ValueError("Invalid part. Choose 'nodes' or 'edges'.")

    def plot(self, path=None):
        fig, ax = ox.plot_graph(
                    self.graph,
                    show=False, close=False,
                    figsize=(256,256)
                )
        if path is not None:
            if isinstance(path, list):
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(self.graph, pos=ox.graph_to_gdfs(self.graph)[0], edgelist=path_edges, edge_color='red', width=2, ax=ax)
            else:
                raise ValueError("Path must be a list of node IDs.")
        plt.show()

    def random_nodes(self, n=1):
        return np.random.choice(
            list(self.graph.nodes),
            size=n,
            replace=False
        )


def main():
    pass

if __name__ == "__main__":
    main()