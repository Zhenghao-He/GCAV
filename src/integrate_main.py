from IntegrateCAV import IntegrateCAV
import numpy as np
import os
import torch.autograd
from align_dim import CAVAutoencoder
from configs import embed_dim, hidden_dims, dropout, dim_align_method, fuse_method, model_to_run, save_dir, num_random_exp,concepts_string, bottlenecks,concepts,target,is_attack
torch.autograd.set_detect_anomaly(True)


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


if __name__ == "__main__":
    overwrite = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if is_attack:
        original_cavs_path = os.path.join(save_dir, model_to_run, "attacked_cavs")
    else:
        original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")

    # concepts_string = "dotted_striped_zigzagged_chequered_honeycombed_scaly"
    # cavs = np.load(os.path.join(save_dir, model_to_run, "original_cavs","cavs_dotted_striped_zigzagged_honeycombed_chequered_scaly.npy"), allow_pickle=True)
    cavs = np.load(os.path.join(original_cavs_path,f"cavs_{concepts_string}.npy"), allow_pickle=True)
    print("traing autoencoders for concept:", concepts_string)
  
    autoencoders = CAVAutoencoder(input_dims=[len(cav[0]) for cav in cavs], embed_dim=embed_dim,hidden_dims=hidden_dims, dropout=dropout, device=device, save_dir=os.path.join(save_dir,model_to_run,"autoencoders",concepts_string), overwrite=False)
    autoencoders.train_autoencoders(cavs=cavs, epochs=30, batch_size=16) #train autoencoder/ we need decoders to reconstruct the cavs for each layer
    # raise ValueError("stop here")
    # import pdb; pdb.set_trace()
    tmp_concepts = concepts
    for concept in tmp_concepts:
        if concept != "striped":
            continue
        print(f"Integrating {concept}")
        concepts = [concept]
        concepts_string = "_".join(concept.replace(" ", "") for concept in concepts)
        # if is_attack:
        #     original_cavs_path = os.path.join(save_dir, model_to_run, "attacked_cavs")
        # else:
        #     original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")
        # original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")
        cavs = np.load(os.path.join(original_cavs_path,f"cavs_{concepts_string}.npy"), allow_pickle=True)
        
        assert len(concepts)==1, "Please provide one concept to integrate"
        integrate_cav = IntegrateCAV(cavs=cavs, device=device, concepts_string=concepts_string,autoencoders=autoencoders,dim_align_method=dim_align_method,num_random_exp=num_random_exp,save_dir=os.path.join(save_dir,model_to_run)).to(device)
        # align before fusion
        # aligned_cavs = integrate_cav.align_with_moco(queue_size=100, momentum=0.999, temperature=0.07, embed_dim=embed_dim,overwrite=overwrite, epochs=1000)
        aligned_cavs = integrate_cav.train(embed_dim=embed_dim,overwrite=False, epochs=1000, batch_size=32, lr=1e-3)
        # aligned_cavs = integrate_cav.align_with_transformer(overwrite=overwrite)
        # import pdb; pdb.set_trace()
        fused_cavs = integrate_cav.fuse(fuse_method=fuse_method,bottlenecks=bottlenecks,concepts=concepts, num_random_exp=num_random_exp, target=target, overwrite=False)

        '''
        TODO:
        1. project the fused cavs to each layer and calculate the tcav scores across different layers for one concept
        '''
        # import pdb; pdb.set_trace()
