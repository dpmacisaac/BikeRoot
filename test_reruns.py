# %%
from bikeroot import BikeNetwork, Evaluation, hmls_solve, ngamobrp_solve, linastar_solve, namoa_solve
import pickle
import os
import os
os.sched_setaffinity(0, {2}) 

# %%

# def run_tests(configs=None):
#     os.sched_setaffinity(0, {2}) 
#     locations = ["Spokane, WA", "San Francisco, CA","Prague", "Aveiro"]
#     path_distances = [500, 1500, 3000, 5000, 10000, 15000, 20000]
#     attempt_file_name = f'data/testing_candidates_4/attempts_4.pkl'
#     attempts = pickle.load(open(attempt_file_name, 'rb')) if os.path.isfile(attempt_file_name) else {}
#     if attempts == {}:
#         for location in locations:
#             attempts[location] = {}
#             for dist in path_distances:
#                 attempts[location][dist] = {'idx': 0, 'completed':0}
#     tries = 0
#     for location in locations:
#         loc_name = location.split(",")[0].lower().replace(" ", "_")
#         candidates = pickle.load(open(f'data/testing_candidates_4/{loc_name}_candidates.pkl', 'rb'))

#         criteria = ['travel distance', 'travel time', 'safety', 'comfort']
#         G = BikeNetwork(location)
#         G.load_graph(f'data/testing_maps_4/{loc_name}_20000_withcost.graphml')
#         G.weights_to_floats(criteria)
#         print(f"Loaded graph for {location} with {len(G.graph.nodes)} nodes and {len(G.graph.edges)} edges.")
#         for dist in candidates:
#             idx = attempts[location][dist]['idx']
#             if idx >= len(candidates[dist]):
#                 print(f"No more candidates for distance {dist} in {location}. Only completed {attempts[location][dist]['completed']} tests.")
#                 continue
#             completed_tests = attempts[location][dist]['completed']
#             if completed_tests >= 4 or dist>9999 and idx>0:
#                 print(f"Skipping distance {dist} in {location} as it has already completed {completed_tests} tests.")
#                 continue
#             if idx >7:
#                 print(f"Skipping distance {dist} in {location} as it has already attempted {idx} tests.")
#                 continue
#             src, dst, lent = candidates[dist][idx]['src'], candidates[dist][idx]['dst'], candidates[dist][idx]['cost']
#             src = str(src)
#             dst = str(dst)
#             print(f"Evaluating from {src} to {dst} for {location} with {lent} in {dist}...")

#             eval_file = f'data/testing_4/{loc_name}_{src}_{dst}_eval.pkl'
            
#             eval = pickle.load(open(eval_file, 'rb')) if os.path.isfile(eval_file) else Evaluation(f'data/testing_maps_2/{loc_name}_20000_withcost.graphml', location, src, dst, criteria, lent=lent,dist_group=dist, G= G.graph, configs=configs)
#             configs = eval.configs if configs is None else configs
#             configs_gen = configs.copy()
#             if dist > 4999:
#                 configs_gen.pop('namoa_tc', None)
#             success = eval.evaluate(configs=configs_gen)
#             if success:
#                 print(f"Completed evaluation for {src} to {dst} in {location}.")
#                 completed_tests += 1

#             idx += 1
#             attempts[location][dist]['idx'] = idx
#             attempts[location][dist]['completed'] = completed_tests
#             pickle.dump(eval, open(eval_file, 'wb'))
#             tries += 1
#             if tries%20 == 0:
#                 pickle.dump(attempts, open(attempt_file_name, 'wb'))
#         pickle.dump(attempts, open(attempt_file_name, 'wb'))

configs = {
    'namoa_tc':{
        'func': namoa_solve,
        'params': {
            'H':'tc',
            'debug': False
        }
    },
    'hmls_125_16_005': {
        'func': hmls_solve,
        'params': {
            'heuristic_skip': {'type': 'ellipse', 'ab': 1.25},
            'heuristic_prune': {'type': 'ratio', 'criteria': 'travel distance', 'alpha': 1.6},
            'heuristic_checkdom': {'type': 'epsilon', 'epsilon': 0.005}
        }
    },
    'hmls_125_16_05': {
        'func': hmls_solve,
        'params': {
            'heuristic_skip': {'type': 'ellipse', 'ab': 1.25},
            'heuristic_prune': {'type': 'ratio', 'criteria': 'travel distance', 'alpha': 1.6},
            'heuristic_checkdom': {'type': 'epsilon', 'epsilon': 0.05}
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
    
    'ngamobrp_20_25_01': {
        'func': ngamobrp_solve,
        'params': {
            'debug': False,
            'n_gen': 20,
            'population_size': 25,
            'mut_prob': 0.01
        }
    },
    'nga2_10_50_01': {
        'func': ngamobrp_solve,
        'params': {
            'debug': False,
            'n_gen': 10,
            'population_size': 50,
            'mut_prob': 0.01,
            'alg':'base'
        }
    },
    'linastar_20': {
        'func': linastar_solve,
        'params': {
            'sol_count': 20,
            'find_all_sols': False,
            'log_level': 'critical'
        }
    },
    'linastar_tc_100': {
        'func': linastar_solve,
        'params': {
            'sol_count': 100,
            'find_all_sols': False,
            'log_level': 'critical',
            'heuristic': 'tc'
        }
    },
    'hmls_110_16_01': {
        'func': hmls_solve,
        'params': {
            'heuristic_skip': {'type': 'ellipse', 'ab': 1.10},
            'heuristic_prune': {'type': 'ratio', 'criteria': 'travel distance', 'alpha': 1.6},
            'heuristic_checkdom': {'type': 'epsilon', 'epsilon': 0.01}
        }
    },
    'ngamobrp_2': {
        'func': ngamobrp_solve,
        'params': {
            'debug': False,
            'n_gen': 60,
            'alg': 'base',
            'surv': 'base'
        }
    }
}
from pathlib import Path
# failures.py  – hand-typed from the console output
FAILED_RUNS: dict[str, list[str]] = {
    # "prague_344905901_306618174_eval.pkl":           ["namoa_tc"],
    "prague_3676311049_1300523301_eval.pkl":         ["hmls_125_16_005"],
    "prague_406383508_1286773192_eval.pkl":          ["hmls_125_16_005"],
    "prague_8197519245_2167454315_eval.pkl":         ["hmls_110_16_01", "hmls_125_16_005"],
    # "san_francisco_6708701281_9777791285_eval.pkl":  ["namoa_tc"],
    "san_francisco_673040400_4924918588_eval.pkl":   ["hmls_110_16_01", "hmls_125_16_005"],
    "san_francisco_7231700331_4550837375_eval.pkl":  ["hmls_125_16_005"],
    "san_francisco_8071218891_308298975_eval.pkl":   ["hmls_125_16_005"],
    # "spokane_12146623728_10159619849_eval.pkl":      ["namoa_tc"],
    "spokane_51056471_8573490448_eval.pkl":          ["hmls_125_16_005"],
}
base_dir = "data/testing_4"
for fname, algs in FAILED_RUNS.items():
    path = Path(base_dir) / fname
    if not path.is_file():
        print(f"[skip] {path} not found on disk")
        continue

    print(f"\n▶  Re-evaluating {fname}  –  {', '.join(algs)}")
    with path.open("rb") as f:
        eval_obj = pickle.load(f)

    # subset configs to the algs that failed here
    cfg_subset = {k: v for k, v in configs.items() if k in algs}
    if not cfg_subset:
        print("   nothing to rerun (config missing)")
        continue
    print(f'{cfg_subset} on {fname}')

    success = eval_obj.re_evaluate(configs=cfg_subset)   # overwrite existing results
    if success:
        print("   ✔  rerun finished without error")
    else:
        print("   ✘  rerun still produced failures")

    # overwrite the pickle so future passes have fresh results
    with path.open("wb") as f:
        pickle.dump(eval_obj, f)


