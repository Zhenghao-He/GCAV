
import numpy as np
import os
from configs import embed_dim, dim_align_method, fuse_method, concepts, bottlenecks, hidden_dims, dropout,save_dir, model_to_run, num_random_exp, concepts_string, fuse_input, is_attack
from align_dim import CAVAutoencoder
# import tensorflow as tf
import torch
import pickle
from tcav.cav import CAV
import glob


if __name__ == "__main__":
    overwrite = True
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")
    if is_attack:
        original_cavs_path = os.path.join(save_dir, model_to_run, "attacked_cavs")
    else:
        original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")
    # cavs = np.load(os.path.join(original_cavs_path,f"cavs_{concepts[0]}.npy"), allow_pickle=True)   
    concepts_string = "dotted_striped_zigzagged_chequered_honeycombed_scaly"
    cavs = np.load(os.path.join(save_dir, model_to_run, "original_cavs","cavs_dotted_striped_zigzagged_honeycombed_chequered_scaly.npy"), allow_pickle=True)
    autoencoders = CAVAutoencoder(input_dims=[len(cav[0]) for cav in cavs], embed_dim=embed_dim,hidden_dims=hidden_dims, dropout=dropout , device=device, save_dir=os.path.join(save_dir,model_to_run,"autoencoders",concepts_string), overwrite=False)
        
        
    decoders =[]
    for layer_idx, bottleneck in enumerate(bottlenecks):
        decoder = autoencoders.load_autoencoder(layer_idx).module.decode
        decoders.append(decoder)
    tmp_concepts = concepts
    for concept in tmp_concepts:
        print(f"reconstructing {concept}")
        concepts = [concept]
        concepts_string = "_".join(concept.replace(" ", "") for concept in concepts)
        
        
        # original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")
        cavs = np.load(os.path.join(original_cavs_path,f"cavs_{concepts_string}.npy"), allow_pickle=True)


        

        fused_cavs = np.load(os.path.join(save_dir, model_to_run,"fuse_model", dim_align_method, fuse_method,concepts_string,fuse_input, f"fused_cavs_{autoencoders.key_params}.npy"), allow_pickle=True)
        reconstructed_save_dir = os.path.join(save_dir, model_to_run, "reconstructed_cavs", dim_align_method, fuse_method, autoencoders.key_params, fuse_input)
        os.makedirs(reconstructed_save_dir, exist_ok=True)
        index = 0
        for layer_idx, bottleneck in enumerate(bottlenecks): # 9
            decoder = decoders[layer_idx]
            concept_idx=0
            for fused_cav in fused_cavs: # concepts 9
                reconstructed = decoder(torch.tensor(fused_cav).to(device)).cpu().detach().numpy()
                random_part = f"random500_{index % num_random_exp}"
                concept = concepts[concept_idx//num_random_exp]
                pattern = f"**/{concept}*{random_part}-{bottleneck}*"
                search_path = os.path.join(original_cavs_path, pattern)
                file = glob.glob(search_path, recursive=True)
                if len(file) > 1:
                    raise ValueError(f"More than one file found for pattern {pattern}")
                file = file[0]
                file_name = os.path.basename(file)
                print(f"Reconstructing {file_name}")
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                data['cavs'][0] = reconstructed
                data['cavs'][1] = -reconstructed
                data['saved_path'] = os.path.join(reconstructed_save_dir, file_name)
                with open(data['saved_path'], 'wb') as f:
                    pickle.dump(data, f)
                print(f"Save Reconstructed {file_name} at {data['saved_path']}")
                index += 1
                concept_idx += 1
                
               
    # aligned_cavs = np.load(os.path.join(save_dir, model_to_run,"align_model", dim_align_method, concepts_string, f"aligned_cavs_{autoencoders.key_params}.npy"), allow_pickle=True)
    # aligned_cavs = np.load(os.path.join(save_dir, model_to_run,"align_model", dim_align_method, concepts_string, f"input_cavs_{autoencoders.key_params}.npy"), allow_pickle=True)
    # aligned_cavs = aligned_cavs[:,:10,:]
    # reconstructed_save_dir = os.path.join(save_dir, model_to_run, "reconstructed_cavs", dim_align_method, fuse_method, autoencoders.key_params)
    # os.makedirs(reconstructed_save_dir, exist_ok=True)
    # index = 0
    # for layer_idx, (bottleneck, aligned_cavs_layer) in enumerate(zip(bottlenecks, aligned_cavs)): # 9
    #     decoder = decoders[layer_idx]
    #     concept_idx=0
    #     for fused_cav in aligned_cavs_layer: # concepts 9
    #         reconstructed = decoder(torch.tensor(fused_cav).to(device)).cpu().detach().numpy()
    #         random_part = f"random500_{index % num_random_exp}"
    #         # import pdb; pdb.set_trace()
    #         concept = concepts[concept_idx//num_random_exp]
    #         pattern = f"**/{concept}*{random_part}-{bottleneck}*"
    #         search_path = os.path.join(original_cavs_path, pattern)
    #         file = glob.glob(search_path, recursive=True)
    #         if len(file) > 1:
    #             raise ValueError(f"More than one file found for pattern {pattern}")
    #         file = file[0]
    #         file_name = os.path.basename(file)
    #         print(f"Reconstructing {file_name}")
    #         with open(file, 'rb') as f:
    #             data = pickle.load(f)
    #         data['cavs'][0] = reconstructed
    #         data['cavs'][1] = -reconstructed
    #         # data['cavs'][0] = np.zeros_like(data['cavs'][0])
    #         # data['cavs'][1] = np.zeros_like(data['cavs'][1])
    #         # import pdb; pdb.set_trace()
    #         data['saved_path'] = os.path.join(reconstructed_save_dir, file_name)
    #         with open(data['saved_path'], 'wb') as f:
    #             pickle.dump(data, f)
    #         print(f"Save Reconstructed {file_name} at {data['saved_path']}")
    #         index += 1
    #         concept_idx += 1
            


    