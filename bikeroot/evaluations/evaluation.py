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

    def __init__(self, graph_name, location, src, dst, criteria,lent,dist_group, configs, G=None):
        self.graph_name = graph_name
        self.location = location
        self.loc_name = location.split(",")[0].lower().replace(" ", "_")
        self.src = src
        self.dst = dst
        self.criteria = criteria
        self.lent = lent
        self.dist_group = dist_group
        self.configs = configs

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

    def evaluate(self, configs= None, redo=False):
        failed_algos = {}
        if configs is None:
            configs = self.configs
        for algo_name, config in configs.items():
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
                signal.alarm(60*5)  # 10 minutes timeout
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
            print(f"Saved results for {algo_name} ({len(costs)} solutions, {end_time - start_time:.2f}s, HV: {hv:.3f})")
            # logging.info(f"Saved results for {algo_name} ({len(costs)} solutions, {end_time - start_time:.2f}s, HV: {hv:.3f})")
        return len(failed_algos) == 0

    def re_evaluate(self, configs= None, redo=False):
        failed_algos = {}
        if configs is None:
            configs = self.configs
        for algo_name, config in configs.items():
            fail = False
            logging.info(f"Evaluating: {algo_name}")
            prev_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _handler)
            start_time = tt.time()
            paths_costs = None
            try:
                signal.alarm(60*10)  # 10 minutes timeout
                paths_costs = config['func'](G=self.graph, src=self.src, dst=self.dst, criteria=self.criteria, **config['params'])
            except Timeout:
                failed_algos[algo_name] = 'Timeout'
                print(f"{algo_name} timed out.")
                fail = True
            except Exception as e:
                failed_algos[algo_name] = {'error': str(e)}
                print(f"{algo_name} crashed: {e}")
                fail = True
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)

            end_time = tt.time(

            )
            if not paths_costs:
                fail = True
                failed_algos[algo_name] = {'error': 'No solutions found'}
                print(f"No solutions for {algo_name}")
            
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
            self.results[algo_name]['trials'].insert(0, trial)  # insert at the beginning to keep the latest trial first
            if algo_name not in self.finished:
                self.finished.append(algo_name)
            print(f"Saved results for {algo_name} ({len(costs)} solutions, {end_time - start_time:.2f}s, HV: {hv:.3f})")
            # logging.info(f"Saved results for {algo_name} ({len(costs)} solutions, {end_time - start_time:.2f}s, HV: {hv:.3f})")
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