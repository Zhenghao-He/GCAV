
import numpy as np
import os
from configs import embed_dim, dim_align_method, fuse_method, concepts, bottlenecks, hidden_dims, dropout,save_dir, model_to_run, num_random_exp, concepts_string, fuse_input,target, concept_map_type, source_dir, activation_dir, mymodel, model_to_run
from align_dim import CAVAutoencoder
# import tensorflow as tf
import torch
import pickle
from tcav.cav import CAV
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tcav.activation_generator as act_gen
import tcav.utils as utils
import tcav.model  as model
import torch.nn.functional as F

def normalize_heatmap(concept_maps):
    min_val = concept_maps.min(axis=(1,2), keepdims=True)
    max_val = concept_maps.max(axis=(1,2), keepdims=True)
    concept_maps = (concept_maps - min_val) / (max_val - min_val + 1e-8)  # 避免除零
    return concept_maps
    


def get_concept_maps(cav, feature_maps):
    num_exs, H, W, C = feature_maps.shape
    # 2. Reshape CAV: (H*W*C,) -> (H, W, C)
    cav = cav.reshape(H, W, C)
    # 3. 计算 Pooled-CAV (C,)，对 CAV 进行 Global Average Pooling
    pooled_cav = cav.mean(axis=(0, 1))  # (C,)
    # 4. 计算每张图的 Concept Map: (num_exs, H, W, C) * (C,) -> (num_exs, H, W)
    concept_maps = np.sum(pooled_cav * feature_maps, axis=-1)  # (num_exs, H, W)
    # 5. 进行 ReLU 操作，去掉负值
    concept_maps = np.maximum(concept_maps, 0)
    # 6. 归一化 Concept Map (逐张图进行归一化)
    # min_val = concept_maps.min(axis=(1,2), keepdims=True)
    # max_val = concept_maps.max(axis=(1,2), keepdims=True)
    # concept_maps = (concept_maps - min_val) / (max_val - min_val + 1e-8)  # 避免除零
    concept_maps = normalize_heatmap(concept_maps)
    return num_exs, concept_maps

def overlay_heatmap_on_image(concept_maps, input_images, output_folder):
    num_exs = len(input_images)

    # 7. 叠加 Concept Map 到原图并保存
    for i in range(num_exs):
        file_path = os.path.join(output_folder, f"overlayed_concept_map_{i+1}.png")
        input_images[i] = (input_images[i] * 255).astype(np.uint8)  # 假设 imgs 是浮点数，范围在 0 到 1 之间

        # 7.1 归一化 Concept Map 到 0-255 并转换为彩色
        heatmap = cv2.applyColorMap((concept_maps[i] * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 7.2 确保 heatmap 和 input_images 大小一致
        if input_images[i].shape[0] != heatmap.shape[0] or input_images[i].shape[1] != heatmap.shape[1]:
            heatmap = cv2.resize(heatmap, (input_images[i].shape[1], input_images[i].shape[0]))

        # 7.3 确保 heatmap 是三通道图像
        if len(heatmap.shape) == 2:  # 如果是单通道
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

        # 7.4 叠加到原图
        overlay = cv2.addWeighted(input_images[i].astype(np.uint8), 0.6, heatmap, 0.4, 0)

        # 7.5 保存叠加后的图片
        cv2.imwrite(file_path, overlay)


    print(f"叠加 Concept Maps 已保存到 {output_folder}")

# def save_overlayed_concept_maps(cav, feature_maps, input_images, output_folder):
#     """
#     计算 Concept Map，并叠加到原图后保存
#     :param cav: CAV (H*W*C,) 需要 reshape 为 (H, W, C)
#     :param feature_maps: 激活值 (num_exs, H, W, C)，多张图片的特征图
#     :param input_images: 原始输入图像 (num_exs, H_o, W_o, 3)，RGB 格式
#     :param output_folder: 保存叠加图像的文件夹路径
#     """

#     num_exs, concept_maps = get_concept_maps(cav, feature_maps)

#     # 7. 叠加 Concept Map 到原图并保存
#     for i in range(num_exs):
#         file_path = os.path.join(output_folder, f"overlayed_concept_map_{i+1}.png")
#         input_images[i] = (input_images[i] * 255).astype(np.uint8)  # 假设 imgs 是浮点数，范围在 0 到 1 之间

#         # 7.1 归一化 Concept Map 到 0-255 并转换为彩色
#         heatmap = cv2.applyColorMap((concept_maps[i] * 255).astype(np.uint8), cv2.COLORMAP_JET)

#         # 7.2 确保 heatmap 和 input_images 大小一致
#         if input_images[i].shape[0] != heatmap.shape[0] or input_images[i].shape[1] != heatmap.shape[1]:
#             heatmap = cv2.resize(heatmap, (input_images[i].shape[1], input_images[i].shape[0]))

#         # 7.3 确保 heatmap 是三通道图像
#         if len(heatmap.shape) == 2:  # 如果是单通道
#             heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

#         # 7.4 叠加到原图
#         overlay = cv2.addWeighted(input_images[i].astype(np.uint8), 0.6, heatmap, 0.4, 0)

#         # 7.5 保存叠加后的图片
#         cv2.imwrite(file_path, overlay)


#     print(f"叠加 Concept Maps 已保存到 {output_folder}")

if __name__ == "__main__":
    overwrite = True

    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    

    # source_dir = '/p/realai/zhenghao/CAVFusion/data'
    # user = 'zhenghao'
    # # the name of the parent directory that results are stored (only if you want to cache)
    # project_name = 'tcav_class_test'
    # working_dir = "/tmp/" + user + '/' + project_name
    # # where activations are stored (only if your act_gen_wrapper does so)
    # activation_dir =  working_dir+ '/activations/'
    # sess = utils.create_session()

    # GRAPH_PATH = source_dir + "/inception5h/tensorflow_inception_graph.pb"


    # LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"

    # mymodel = model.GoogleNetWrapper_public(sess,
    #                                         GRAPH_PATH,
    #                                         LABEL_PATH)
    act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)
    class_examples = act_generator.get_examples_for_concept(target) 
    class_acts_layer = {} # (layer_n, num_exs, acts)
    acts = act_generator.process_and_load_activations(bottlenecks, [target])
    for bottleneck in bottlenecks:
        acts_instance = acts[target][bottleneck]
        class_acts_layer[bottleneck] = acts_instance # (num_exs, H, W, D)

    #  act = np.expand_dims(class_acts_layer[bottleneck][i], 0)
    # import pdb; pdb.set_trace()

    original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")
    cavs = np.load(os.path.join(original_cavs_path,f"cavs_{concepts_string}.npy"), allow_pickle=True)
    autoencoders = CAVAutoencoder(input_dims=[len(cav[0]) for cav in cavs], embed_dim=embed_dim,hidden_dims=hidden_dims, dropout=dropout , device=device, save_dir=os.path.join(save_dir,model_to_run,"autoencoders",concepts_string), overwrite=False)
    reconstructed_save_dir = os.path.join(save_dir, model_to_run, "reconstructed_cavs", dim_align_method, fuse_method, autoencoders.key_params, fuse_input)
    if not os.path.exists(reconstructed_save_dir):
        raise ValueError(f"Reconstructed CAVs not found at {reconstructed_save_dir}")

    assert len(concepts) == 1, "Only one concept is supported"
    concept = concepts[0] # test meaningfull concept
    output_folder = os.path.join(save_dir, model_to_run, "concept_maps", concept_map_type, target, concept)
    os.makedirs(output_folder, exist_ok=True)
    
    for layer_idx, bottleneck in enumerate(bottlenecks):
        activate = class_acts_layer[bottleneck]
        pattern = f"{concept}-random500_0-{bottleneck}-linear-0.1.pkl"
        if concept_map_type == "original_cavs":
            file = os.path.join(original_cavs_path, pattern)
        elif concept_map_type == "reconstructed_cavs":
            file = os.path.join(reconstructed_save_dir, pattern)
        with open(file, 'rb') as f:
            data = pickle.load(f)
        cav = data['cavs'][0]
        save_folder = os.path.join(output_folder, bottleneck)
        os.makedirs(save_folder, exist_ok=True)
        num_exs, concept_maps = get_concept_maps(cav, activate)
        overlay_heatmap_on_image(concept_maps=concept_maps, input_images=class_examples, output_folder=save_folder)


    global_heatmap = np.zeros((50,28,28))
    target_size = (28, 28)
    weights_list = []
    for layer_idx, bottleneck in enumerate(bottlenecks):
        activate = class_acts_layer[bottleneck]
        pattern = f"{concept}-random500_0-{bottleneck}-linear-0.1.pkl"
        ori_file = os.path.join(original_cavs_path, pattern)
        recon_file = os.path.join(reconstructed_save_dir, pattern)
        with open(ori_file, 'rb') as f:
            ori_data = pickle.load(f)
        ori_cav = ori_data['cavs'][0]
        with open(recon_file, 'rb') as f:
            recon_data = pickle.load(f)
        recon_cav = recon_data['cavs'][0]

        weight = F.cosine_similarity(torch.tensor(ori_cav), torch.tensor(recon_cav), dim=0)
        weights_list.append((bottleneck, weight.item()))
        if concept_map_type == "original_cavs":
            _, concept_map = get_concept_maps(ori_cav, activate)
        elif concept_map_type == "reconstructed_cavs":
            _, concept_map = get_concept_maps(recon_cav, activate)
        if concept_map.shape[-2:] != (target_size[1], target_size[0]):
            resized_concept_map = np.zeros((concept_map.shape[0], target_size[1], target_size[0]))
            for i in range(concept_map.shape[0]):
                # cv2.resize 的目标尺寸为 (width, height)
                resized_concept_map[i] = cv2.resize(concept_map[i], target_size)
            concept_map = resized_concept_map

        global_heatmap += concept_map * 1
        # global_heatmap += concept_map * weight.item()
        
    
    global_heatmap = normalize_heatmap(global_heatmap)
    save_folder = os.path.join(output_folder, "global")
    os.makedirs(save_folder, exist_ok=True)
    overlay_heatmap_on_image(global_heatmap, class_examples, save_folder)
    weight_txt_path = os.path.join(output_folder, "weights.txt")
    with open(weight_txt_path, "w") as f:
        for bottleneck, w in weights_list:
            f.write(f"{bottleneck}: {w}\n")

    print(f"权重已保存到 {weight_txt_path}")
