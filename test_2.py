# %%
from bikeroot import BikeNetwork, Evaluation, hmls_solve, ngamobrp_solve, linastar_solve, namoa_solve
import pickle
import os
import os
os.sched_setaffinity(0, {2}) 

# %%

def run_tests(configs=None):
    os.sched_setaffinity(0, {2}) 
    locations = ["Spokane, WA", "San Francisco, CA","Prague", "Aveiro"] 
    index = 0
    for location in locations:
        loc_name = location.split(",")[0].lower().replace(" ", "_")
        candidates = pickle.load(open(f'data/testing_candidates_2/{loc_name}_candidates.pkl', 'rb'))


        criteria = ['travel distance', 'travel time', 'safety', 'comfort']
        G = BikeNetwork(location)
        G.load_graph(f'data/testing_maps/{loc_name}_20000_withcost.graphml')
        G.weights_to_floats(criteria)
        print(f"Loaded graph for {location} with {len(G.graph.nodes)} nodes and {len(G.graph.edges)} edges.")

        for dist in candidates:
            idx = 0
            completed_tests = 0
            testing = True
            while testing:
                if idx >= len(candidates[dist]):
                    testing = False
                    print(f"No more candidates for distance {dist} in {location}. Only completed {completed_tests} tests.")
                    continue

                src, dst, lent = candidates[dist][idx]['src'], candidates[dist][idx]['dst'], candidates[dist][idx]['cost']
                src = str(src)
                dst = str(dst)
                print(f"Evaluating from {src} to {dst} for {location} with {lent}...")
                eval_file = f'data/testing_3/{loc_name}_{src}_{dst}_eval.pkl'
                eval = pickle.load(open(eval_file, 'rb')) if os.path.isfile(eval_file) else Evaluation(f'data/testing_maps/{loc_name}_20000_withcost.graphml', location, src, dst, criteria, lent=lent,dist_group=dist, G= G.graph, configs=configs)
                configs = eval.configs if configs is None else configs
                if len(eval.finished) == len(configs):
                    print(f"Skipping {src} to {dst} as it is already finished.")
                    idx += 1
                    success = True
                else:
                    success = eval.evaluate(configs=configs)
                if success:
                    completed_tests += 1

                idx += 1
                if dist > 9999 and completed_tests >= 1:
                    testing = False
                if completed_tests >= 4 or idx > 10:
                    testing = False
                pickle.dump(eval, open(eval_file, 'wb'))

        # for src,dst,lent in candidates:
        #     # if index < index_finished:
        #     #     print(f"Skipping {index} as it is already finished.")
        #     #     index += 1
        #     #     continue
        #     src = str(src)
        #     dst = str(dst)
        #     print(f"Evaluating from {src} to {dst} for {location} with {lent}...")
        #     eval = Evaluation(f'data/testing_maps/{loc_name}_20000_withcost.graphml', location, src, dst, criteria, G= G.graph)
            
        #     eval.evaluate()
        #     index += 1
        #     pickle.dump(index, open(f'data/testing_candidates/{loc_name}_index_finished.pkl', 'wb'))
        #     print(f"Finished evaluation for {src} to {dst}. Current index: {index}")

# %%

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
    'hmls_125_16_01': {
        'func': hmls_solve,
        'params': {
            'heuristic_skip': {'type': 'ellipse', 'ab': 1.25},
            'heuristic_prune': {'type': 'ratio', 'criteria': 'travel distance', 'alpha': 1.6},
            'heuristic_checkdom': {'type': 'epsilon', 'epsilon': 0.01}
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
    
    'ngamobrp_10_80_01': {
        'func': ngamobrp_solve,
        'params': {
            'debug': False,
            'n_gen': 10,
            'population_size': 80,
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
run_tests(configs=configs)


# %%



