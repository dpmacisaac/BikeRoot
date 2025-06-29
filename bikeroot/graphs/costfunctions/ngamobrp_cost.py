from ..cost import Cost
from .. import utils
import math


class NgaMobrp_Cost(Cost):
    
    def __init__(self):
        # criteria: travel distance, travel time, safety, and comfort

        '''
        Notes on how it works:
            - Each cost function has two componenets: an edge cost and a node cost
            - The paper formats this as a single function
            - Since edge + node combo isn't known initially, we will calculate the edge and node costs seperately
            - Then the search algorithm will calculate the cost when considering a new route by adding the two
        '''


        self.G = None
        # precalculating some values
        self.slope = None
        self.c_slope = None
        self.v_calc = None
        self.travel_distance = None
        self.travel_time = None
        self.safety = None
        self.comfort = None
        self.criteria_labels = ['travel distance', 'travel time', 'safety', 'comfort']
    
    def has_node_data(self):
        return True

    def _calculate_distance(self, edge):
        coords_u, coords_v = utils.get_edge_coords(self.G, edge)
        return utils.haversine_distance(coords_u, coords_v)
    
    def _calculate_time(self, edge):
        numerator = self.travel_distance
        denominator = self.v_calc * self._ss(edge)
        if denominator == 0: denominator = 1
        return numerator / denominator
    
    def _calculate_safety(self, edge):
        return self.travel_distance * self._edge_fs(edge) * self.c_slope
    
    def _calculate_comfort(self, edge):
        return self.travel_distance * self._edge_fc(edge) * self.c_slope

    def _slope(self, edge):
        return edge[2]['grade']
    
    def _ss(self, edge):
        if 'surface' in edge[2]:
            if isinstance(edge[2]['surface'], list):
                worst_case = float("inf")
                for type in edge[2]['surface']:
                    if type in self.edge_features['surface']:
                        worst_case = min(worst_case, self.edge_features['surface'][type]['ss'])

                return 0.85 if worst_case is float("inf") else worst_case
            
            elif edge[2]['surface'] in self.edge_features['surface']:
                return self.edge_features['surface'][edge[2]['surface']]['ss']
        
        return 0.85
    
    def _edge_fs(self, edge):
        worst_case = -1
        for key in ['highway', 'bicycle', 'junction', 'bridge', 'crossing']:
            if key in edge[2]:
                if isinstance(edge[2][key], list):
                    for type in edge[2][key]:
                        if type in self.edge_features[key]:
                            worst_case = max(worst_case, self.edge_features[key][type]['edge_fs'])
                        
                elif edge[2][key] in self.edge_features[key]:
                    worst_case = max(worst_case, self.edge_features[key][edge[2][key]]['edge_fs'])
        return worst_case if worst_case != -1 else 1
    
    def _edge_fc(self, edge):
        worst_case = -1
        for key in ['highway', 'surface', 'bicycle', 'junction', 'bridge', 'crossing']:
            if key in edge[2]:
                if isinstance(edge[2][key], list):
                    for type in edge[2][key]:
                        if type in self.edge_features[key]:
                            edge_fc = self.edge_features[key][type]['edge_fc']
                            if edge_fc is not None:
                                worst_case = max(worst_case, edge_fc)
                        
                elif edge[2][key] in self.edge_features[key]:
                    edge_fc = self.edge_features[key][edge[2][key]]['edge_fc']
                    if edge_fc is not None:
                        worst_case = max(worst_case, edge_fc)
        return worst_case if worst_case != -1 else 1

    def _c_slope(self, slope):
        return 2 - (self.v_calc/self._V_calculation(-1))

    def _V_calculation(self,sl):
        if sl < -0.92:
            return 7.582*math.exp(0.1072*sl)
        elif sl <= 6:
            return 5.787*math.exp(-0.1880*sl)
        elif sl <= 10:
            return 0.833
        else:
            return 0


    def get_edge_cost(self, edge, graph):
        self.G = graph

        # precalculating some values
        self.slope = self._slope(edge)
        self.v_calc = self._V_calculation(self.slope)
        self.c_slope = self._c_slope(self.slope)

        self.travel_distance = self._calculate_distance(edge)
        self.travel_time = self._calculate_time(edge)
        self.safety = self._calculate_safety(edge)
        self.comfort = self._calculate_comfort(edge)
        

        features = {'travel distance': self.travel_distance,
                    'travel time': self.travel_time,
                    'safety': self.safety,
                    'comfort': self.comfort,
                    'length':self.travel_distance,
                   }
        
        return features

    def _calculate_node_penalty(self,criteria, node):
        worst_case = 0
        for key in ['highway', 'crossing', 'traffic_calming']:
            if key in node[1]:
                if isinstance(node[1][key], list):
                    for type in node[1][key]:
                        if type in self.node_features[key]:
                            val = self.node_features[key][type][criteria]
                            if val is not None:
                                worst_case = max(worst_case, val)
                
                elif node[1][key] in self.node_features[key]:
                    val = self.node_features[key][node[1][key]][criteria]
                    if val is not None:
                        worst_case = max(worst_case, val)
        return worst_case
    

    def get_node_cost(self, node, graph):
        self.G = graph

        self.time_penalty = self._calculate_node_penalty('f', node)
        self.safety_penalty = self._calculate_node_penalty('node_fc',node)
        self.comfort_penalty = self._calculate_node_penalty('node_fs',node)
        self.lat = node[1]['y']
        self.lon = node[1]['x']

        features = {'travel distance': 0.0,
                    'travel time': self.time_penalty,
                    'safety': self.safety_penalty,
                    'comfort': self.comfort_penalty,
                    'y': self.lat,
                    'x': self.lon
                   }
        
        return features
    
    node_features = {
        'highway': {
            'steps': {'f': 20, 'node_fc': 300, 'node_fs': None},
            'give_way': {'f': 2, 'node_fc': 100, 'node_fs': 100},
            'crossing': {'f': 4, 'node_fc': 100, 'node_fs': 120},
            'traffic_signals': {'f': 2, 'node_fc': 100, 'node_fs': 70},
            'elevator': {'f': 15, 'node_fc': 150, 'node_fs': None},
            'stop': {'f': 4, 'node_fc': 300, 'node_fs': None}
        },
        'crossing':{
            'zebra': {'f': 3, 'node_fc': 160, 'node_fs': None},
            'traffic_signals': {'f': 3, 'node_fc': 160, 'node_fs': None},
            'uncontrolled': {'f': 3, 'node_fc': 160, 'node_fs': 120}
        },
        'traffic_calming':{
            'bump': {'f': 2, 'node_fc': 140, 'node_fs': 5},
            'table': {'f': 2, 'node_fc': 140, 'node_fs': 5}
        }
    }
    edge_features = {
        'highway': {
            'cycleway': {'ss': None, 'edge_fc': 0.57, 'edge_fs': 0.15},
            'steps': {'ss': None, 'edge_fc': 1.85, 'edge_fs': 1.85},
            'footway': {'ss': None, 'edge_fc': None, 'edge_fs': 0.81},
            'pedestrian': {'ss': None, 'edge_fc': None, 'edge_fs': 0.81},
            'track': {'ss': None, 'edge_fc': None, 'edge_fs': 0.81},
            'path': {'ss': None, 'edge_fc': None, 'edge_fs': 0.81},
            'primary': {'ss': None, 'edge_fc': None, 'edge_fs': 1.19},
        },
        'surface': {
            'ground': {'ss': 0.90, 'edge_fc': 1.40, 'edge_fs': None},
            'wood': {'ss': 0.80, 'edge_fc': 2.00, 'edge_fs': None},
            'sand': {'ss': 0.75, 'edge_fc': 2.00, 'edge_fs': None},
            'sett': {'ss': 0.85, 'edge_fc': 1.65, 'edge_fs': None},
            'gravel': {'ss': 0.75, 'edge_fc': 1.70, 'edge_fs': None},
            'paving_stones': {'ss': 0.85, 'edge_fc': 1.50, 'edge_fs': None},
            'unpaved': {'ss': 0.75, 'edge_fc': 1.50, 'edge_fs': None},
            'cobblestone': {'ss': 0.80, 'edge_fc': 1.50, 'edge_fs': None},
            'concrete': {'ss': 0.95, 'edge_fc': 1.00, 'edge_fs': None},
            'asphalt': {'ss': 1.00, 'edge_fc': 1.00, 'edge_fs': None},
        },
        'bicycle': {
            'designated': {'ss': None, 'edge_fc': 0.57, 'edge_fs': 0.43},
            'yes': {'ss': None, 'edge_fc': 1.00, 'edge_fs': 1.00},
        },
        'junction': {
            'roundabout': {'ss': None, 'edge_fc': 5.00, 'edge_fs': 3.00},
        },
        'bridge': {
            'yes': {'ss': None, 'edge_fc': 7.00, 'edge_fs': 1.19},
            'movable': {'ss': None, 'edge_fc': 7.00, 'edge_fs': 1.19},
            'suspension': {'ss': None, 'edge_fc': 7.00, 'edge_fs': 1.19},
        },
        'crossing': {
            'zebra': {'ss': None, 'edge_fc': 2.00, 'edge_fs': 1.30},
            'marked': {'ss': None, 'edge_fc': 2.00, 'edge_fs': 1.30},
            'island': {'ss': None, 'edge_fc': 2.00, 'edge_fs': 1.40},
            'traffic_signals': {'ss': None, 'edge_fc': 2.00, 'edge_fs': 1.50},
        }
    }