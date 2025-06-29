"""Evaluation framework for multi-objective bike routing algorithms.
   Applies configurable algorithms to a graph and stores performance results.
"""
import os
import time as tt
import signal
import pickle
import logging
import numpy as np
from pymoo.indicators.hv import HV
from bikeroot import (
    BikeNetwork, CostApplicator, Cost, NgaMobrp_Cost, alg_utils,
    eval_utils, NGAMOBRPProblem,
    biobjective_label_setting_solve, biobjective_label_correcting_solve,
    hmls_solve, ngamobrp_solve, nga3mobrp_solve, moedbrp_solve
)
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# For CPU pinning (Linux only)
try:
    os.sched_setaffinity(0, {2})  # pin to core 2
except AttributeError:
    logging.warning("CPU pinning not supported on this system.")

class Timeout(Exception): pass
def _handler(signum, frame): raise Timeout()

class Evaluation:
    # Algorithm configurations
    configs = {
        'hmls_125_16_005': {
            'func': hmls_solve,
            'params': {
                'heuristic_skip': {'type': 'ellipse', 'ab': 1.25},
                'heuristic_prune': {'type': 'ratio', 'criteria': 'travel distance', 'alpha': 1.6},
                'heuristic_checkdom': {'type': 'epsilon', 'epsilon': 0.005}
            }
        },
        'ngamobrp_10_50_01': {
            'func': ngamobrp_solve,
            'params': {
                'debug': False,
                'n_gen': 10,
                'population_size': 50,
                'mut_prob': 0.01
            }
        },
        # 'ngamobrp_2': {
        #     'func': ngamobrp_solve,
        #     'params': {
        #         'debug': False,
        #         'n_gen': 60,
        #         'alg': 'base',
        #         'surv': 'base'
        #     }
        # }
    }

    def __init__(self, graph_name, location, src, dst, criteria,lent,dist_group, G=None):
        self.graph_name = graph_name
        self.location = location
        self.loc_name = location.split(",")[0].lower().replace(" ", "_")
        self.src = src
        self.dst = dst
        self.criteria = criteria
        self.lent = lent
        self.dist_group = dist_group

        if G:
            self.graph = G
        else:
            bike_net = BikeNetwork(location)
            bike_net.load_graph(graph_name)
            bike_net.weights_to_floats(criteria)
            self.graph = bike_net.graph

        self.hypervolume_ref_point = self._set_hypervolume_ref_point()
        self.results = {}
        self.finished = []
    def __setstate__(self, state):
        # state is whatever was pickled
        self.__dict__.update(state)
        # supply defaults that might be missing

        if 'finished' not in self.__dict__:
            self.finished = [alg_name for alg_name in self.results.keys()]
    def _set_hypervolume_ref_point(self):
        nga = NGAMOBRPProblem(self.graph, self.src, self.dst, criteria=self.criteria)
        ref_costs = []
        for i, c in enumerate(self.criteria):
            path = nga.a_star(self.src, self.dst, i)
            cost_vector = [alg_utils.total_path_cost(self.graph, path, k) if path else -1 for k in self.criteria]
            ref_costs.append(cost_vector)
        max_vals = np.max(np.array(ref_costs), axis=0)
        return max_vals.astype(float)

    def evaluate(self, redo=False):
        failed_algos = {}
        for algo_name, config in self.configs.items():
            fail = False
            logging.info(f"Evaluating: {algo_name}")
            prev_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _handler)
            start_time = tt.time()
            paths_costs = None
            if algo_name in self.results and not redo:
                logging.info(f"Skipping {algo_name}, already evaluated.")
                continue
            try:
                signal.alarm(600)  # 10 minutes timeout
                paths_costs = config['func'](G=self.graph, src=self.src, dst=self.dst, criteria=self.criteria, **config['params'])
            except Timeout:
                failed_algos[algo_name] = 'Timeout'
                logging.warning(f"{algo_name} timed out.")
                fail = True
            except Exception as e:
                failed_algos[algo_name] = {'error': str(e)}
                logging.exception(f"{algo_name} crashed: {e}")
                fail = True
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)

            end_time = tt.time()
            if not paths_costs:
                fail = True
                failed_algos[algo_name] = {'error': 'No solutions found'}
                logging.warning(f"No solutions for {algo_name}")
            
            if fail:
                paths = []
                costs = []
                time = -1
                hv = -1
            else:
                paths, costs = Evaluation.paths_costs_split(paths_costs)
                paths, costs = self.remove_dominated_solutions(paths, costs)
                hv = self.calculate_hypervolume(costs)
                time = end_time - start_time

            trial = {
                'paths': paths,
                'costs': costs,
                'time': time,
                'hypervolume': hv,
                'count': len(costs),
                'fail': None if not fail else failed_algos[algo_name]
            }

            self.results.setdefault(algo_name, {'config': config, 'trials': []})
            self.results[algo_name]['trials'].append(trial)
            if algo_name not in self.finished:
                self.finished.append(algo_name)
            logging.info(f"Saved results for {algo_name} ({len(costs)} solutions, {end_time - start_time:.2f}s, HV: {hv:.3f})")
        return len(failed_algos) == 0

    @staticmethod
    def paths_costs_split(paths_costs):
        paths, costs = zip(*paths_costs)
        return list(paths), list(costs)

    def calculate_hypervolume(self, costs):
        F = np.array([[c[k] for k in self.criteria] for c in costs], dtype=float)
        hv = HV(ref_point=self.hypervolume_ref_point)
        value = hv.do(F)
        # logging.info(f"Hypervolume: {value:.3f}")
        return value
    
    def remove_dominated_solutions(self, paths, costs):
        """Removes dominated solutions based on the criteria. Returns only non-dominated ones."""
        if not costs:
            return paths, costs

        # Convert cost dicts to a 2D array for dominance checking
        F = np.array([[c[k] for k in self.criteria] for c in costs], dtype=float)

        # Find indices of non-dominated solutions
        nd_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)

        # Filter and return only the non-dominated paths and costs
        nd_paths = [paths[i] for i in nd_indices]
        nd_costs = [costs[i] for i in nd_indices]
        return nd_paths, nd_costs

class Result:
    pass

# import numpy as np
# from pymoo.indicators.hv import HV
# from bikeroot import BikeNetwork, CostApplicator, Cost, NgaMobrp_Cost, alg_utils, eval_utils, NGAMOBRPProblem
# import pickle
# import os
# from bikeroot import biobjective_label_setting_solve, biobjective_label_correcting_solve, hmls_solve, ngamobrp_solve, nga3mobrp_solve, moedbrp_solve
# import time
# import signal

# class Timeout(Exception): pass

# def _handler(signum, frame):
#     raise Timeout()

# class Evaluation():
#     configs = {
#         'hmls': {
#             'func':hmls_solve,
#             'params':{
#                 'heuristic_skip': {'type': 'ellipse', 'ab':1.6},
#                 'heuristic_prune': {'type':'ratio', 'criteria': 'travel distance', 'alpha':1.6}  ,
#                 'heuristic_checkdom': {'type': 'epsilon', 'epsilon': 0.005}
#                 }
#             },
#         'ngamobrp': {
#             'func': ngamobrp_solve,
#             'params': {
#                 'debug': False,
#                 'n_gen':60
#             }
#         },
#         'ngamobrp_2': {
#             'func': ngamobrp_solve,
#             'params': {
#                 'debug': False,
#                 'n_gen':60,
#                 'alg' : 'base',
#                 'surv': 'base'
#             }
#         },
#         # 'nga3mobrp': {
#         #     'func': nga3mobrp_solve,
#         #     'params': {
#         #         'debug': False
#         #     }
#         # },
#         # 'moedbrp': {
#         #     'func': moedbrp_solve,
#         #     'params': {
#         #         'debug': False
#         #     }
#         # }

#     }

#     def __init__(self, graph_name, location, src, dst, criteria, G = None):
#         self.graph_name = graph_name
#         self.location = location
#         self.loc_name = location.split(",")[0].lower().replace(" ", "_")
#         self.src = src
#         self.dst = dst
#         self.criteria = criteria

#         if G is not None:
#             self.graph = G
#         else:
#             bike_net = BikeNetwork(location)
#             bike_net.load_graph(graph_name)
#             bike_net.weights_to_floats(self.criteria)
#             self.graph = bike_net.graph
        
#         self.hypervolume_ref_point = self._set_hypervolume_ref_point()
#         criteria_str = "_".join(self.criteria).replace(" ", "_").lower()
#         self.result_file = f'data/testing/{self.loc_name}_{src}_{dst}_{criteria_str}.pkl'
#         self.results = pickle.load(open(self.result_file, 'rb')) if os.path.exists(self.result_file) else {}

#     def _set_hypervolume_ref_point(self):
#         nga = NGAMOBRPProblem(self.graph, self.src, self.dst, criteria=self.criteria)
#         criteria_costs = {c: [] for c in self.criteria}
#         for indx, c in enumerate(self.criteria):
#             path = nga.a_star(self.src, self.dst, indx)
#             for k in self.criteria:
#                 cost_k = alg_utils.total_path_cost(self.graph, path, weight=k) if path else -1
#                 criteria_costs[k].append(cost_k)
#         criteria_refpoints = {c: -1 for c in self.criteria}
#         for c in self.criteria:
#             criteria_refpoints[c] = max(criteria_costs[c]) if criteria_costs[c] else -1
#         ref_point = np.array([criteria_refpoints[k] for k in self.criteria], dtype=float)
#         return ref_point
    
#     def evaluate(self):
#         # result = {'config': None, 'trials':[]}
#         # trial = {'paths':[], 'costs':[], 'time': -1, 'hypervolume': -1, 'ndsolutions':-1}

#         for algo_name, config in self.configs.items():
#             print(f"Evaluating {algo_name} algorithm...")
#             prev_handler = signal.getsignal(signal.SIGALRM)
#             signal.signal(signal.SIGALRM, _handler)
#              # Set a timeout of 10 minutes for each algorithm evaluation 
#             start_time = time.time()
#             try:
#                 signal.alarm(480)
#                 paths_costs = config['func'](G=self.graph, src=self.src, dst=self.dst, criteria=self.criteria, **config['params'])
#             except Timeout:
#                 print(f"{algo_name} timed out after 10 min; skipping.  {algo_name} algorithm with config: {config['params']}, src: {self.src}, dst: {self.dst}, location: {self.location}")
#                 continue  
#             except:
#                 print(f"Error evaluating {algo_name} algorithm with config: {config['params']}, src: {self.src}, dst: {self.dst}, location: {self.location}")
#                 continue
#             finally:
#                 signal.alarm(0)                  # **always** cancel alarm
#                 signal.signal(signal.SIGALRM, prev_handler)
#             end_time = time.time()
#             if paths_costs is None or len(paths_costs) == 0:
#                 print(f"No paths found for {algo_name} algorithm with config: {config['params']}, src: {self.src}, dst: {self.dst}, location: {self.location}")
#                 continue
#             paths, costs = Evaluation.paths_costs_split(paths_costs)

#             if algo_name not in self.results:
#                 self.results[algo_name] = {'config': config, 'trials':[]}
#             print(f'solution count: {len(costs)}, time taken: {end_time - start_time:.2f} seconds')
#             trial_result = {
#                 'paths': paths,
#                 'costs': costs,
#                 'time': end_time - start_time,
#                 'hypervolume': self.calculate_hypervolume(costs)
#                 }
#             self.results[algo_name]['trials'].append(trial_result)
#             pickle.dump(self.results, open(self.result_file,'wb'))

#     def paths_costs_split(paths_costs):
#         paths = []
#         costs = []
#         for path, cost in paths_costs:
#             paths.append(path)
#             costs.append(cost)
#         return paths, costs
    
#     def calculate_hypervolume(self, costs):

#         F = np.array([[c[k] for k in self.criteria] for c in costs], dtype=float)
#         hv = HV(ref_point=self.hypervolume_ref_point)
#         value = hv.do(F)

#         print(f"Hypervolume = {value:.3f}")
#         return value
