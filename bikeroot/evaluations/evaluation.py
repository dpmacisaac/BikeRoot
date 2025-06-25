class Result():
    def __init__(self, graph_name, location, distance, algorithm, src, dst, criteria, params={}):
        self.graph = graph_name
        self.location = location
        self.distance = distance
        self.algorithm = algorithm
        self.params = params
        self.src = src
        self.dst = dst
        self.criteria = criteria
        self.path_sets = []
        self.costs = []
        self.times = []
        self.hypervolumes = []
        self.nadir_point = None
        self.hypervolume_ref_point = None

    def add_result(self, paths, costs, time):
        """
        Add a new result to the result set.
        :param paths: List of paths (each path is a list of nodes)
        :param costs: List of costs corresponding to each path
        :param times: List of times corresponding to each path
        """
        self.path_sets.append(paths)
        self.costs.append(costs)
        self.times.append(time)
        hypervolume = self.calculate_hypervolume(costs)
        if hypervolume is not None:
            self.hypervolumes.append(hypervolume)
    
    def calculate_hypervolume(self, costs):
        if not self.hypervolume_ref_point:
            print("Hypervolume reference point not set. Please set it before evaluating hypervolume.")
            return None
        return None

    def __str__(self):
        return f"Result(graph={self.graph}, location={self.location}, distance={self.distance}, algorithm={self.algorithm}, src={self.src}, dst={self.dst}, criteria={self.criteria})"