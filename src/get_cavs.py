from tcav.cav import get_or_train_cav
import pickle
from configs import concepts, bottlenecks, model_to_run, save_dir, concepts_string, num_random_exp, is_attack
import glob
import numpy as np
import os
import random
import itertools
def get_cavs(concepts, bottlenecks, cav_dir=None,num_random_exp=10):
    cavs_instances = []
    for bottleneck in bottlenecks:
        cavs_layer = []
        for concept in concepts:
            for i in range(num_random_exp):
                pattern = f"**/{concept}*random500_{i}-*{bottleneck}*"
                search_path = os.path.join(cav_dir, pattern)
                files = glob.glob(search_path, recursive=True)
                for file in files:
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    cav = data['cavs'][0] # use the first CAV
                    # import pdb;pdb.set_trace()
                    cavs_layer.append(cav)
        cavs_instances.append(cavs_layer)
    return cavs_instances

def get_random_cavs(bottlenecks, cav_dir=None,num_random_exp=10):
    cavs_instances = []
    pairs = list(itertools.combinations(range(num_random_exp), 2))
    for bottleneck in bottlenecks:
        cavs_layer = []
        print(f"Getting random cavs for {bottleneck}")
        for pair in pairs:
            pattern = f"**/random500_{pair[0]}-random500_{pair[1]}-{bottleneck}*"
            search_path = os.path.join(cav_dir, pattern)
            files = glob.glob(search_path, recursive=True)
            if len(files) == 0:
                raise ValueError(f"No file found for pattern {pattern}")
            elif len(files) > 1:
                raise ValueError(f"More than one file found for pattern {pattern}")
            with open(files[0], 'rb') as f:
                data = pickle.load(f)
            cav = data['cavs'][0]
            cavs_layer.append(cav)
        cavs_instances.append(cavs_layer)
    return cavs_instances
    #     while cnt < num_random_exp:
    #         print(cnt)
    #         nums = random.sample(range(0, num_random_exp), 2)
    #         pattern = f"**/random500_{nums[0]}-random500_{nums[1]}-{bottleneck}*"
    #         print(pattern)
    #         search_path = os.path.join(cav_dir, pattern)
    #         files = glob.glob(search_path, recursive=True)
    #         if len(files) == 0:
    #             continue
    #         elif len(files) > 1:
    #             raise ValueError(f"More than one file found for pattern {pattern}")
    #         for file in files:
    #             with open(file, 'rb') as f:
    #                 data = pickle.load(f)
    #             cav = data['cavs'][0] # use the first CAV
    #             cavs_layer.append(cav)
    #         cavs_instances.append(cavs_layer)
    #         cnt+=1
    # return cavs_instances

if __name__ == "__main__":
    
    tmp_concepts = concepts
    if is_attack:
        original_cavs_path = os.path.join(save_dir, model_to_run, "attacked_cavs")
    else:
        original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")
    cavs = get_cavs(concepts, bottlenecks, cav_dir=original_cavs_path, num_random_exp=num_random_exp)
    all_cavs = np.array(cavs, dtype=object)
    np.save(os.path.join(original_cavs_path, f"cavs_attack.npy"), all_cavs)
    # np.save(os.path.join(original_cavs_path, f"cavs_{concepts_string}.npy"), all_cavs)
    # import pdb;pdb.set_trace()
    for concept in tmp_concepts:
        print(f"getting cavs for {concept}")
        concepts = [concept]
        concepts_string = "_".join(concept.replace(" ", "") for concept in concepts)
        
        assert num_random_exp >= 10, "num_random_exp should be at least 10"
        cavs = get_cavs(concepts, bottlenecks, cav_dir=original_cavs_path, num_random_exp=num_random_exp)
        # os.makedirs(os.path.join(original_cavs_path, "original_cavs"), exist_ok=True)
        cavs_array = np.array(cavs, dtype=object)
        random_cavs = get_random_cavs(bottlenecks, cav_dir=original_cavs_path, num_random_exp=num_random_exp)
        cavs_array = np.concatenate(([cavs_array, np.array(random_cavs, dtype=object)]), axis = 1)
        np.save(os.path.join(original_cavs_path, f"cavs_{concepts_string}.npy"), cavs_array)

    import pdb;pdb.set_trace()
