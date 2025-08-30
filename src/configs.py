# this is a regularizer penalty parameter for linear classifier to get CAVs. 
import torch
import tcav.utils as utils
import tcav.model  as model
import os
alphas = [0.1]   
num_random_exp=2
if num_random_exp != 10:
    for i in range(10):
        print("WARNING: num_random_exp is not 10, please check the code")
# target = 'cab'
# concepts = ["car"]

# target = 'groom'
# concepts = ["human"]

# target = 'robin'
# concepts = ["bird"]

# target = 'zebra'  
# concepts = ["striped"]
# concepts = ["perturbed_images_mixed5a"]  #
# concepts = ["dotted_to_striped_mixed3a_light","dotted_to_striped_mixed3b_light","dotted_to_striped_mixed4a_light","dotted_to_striped_mixed4b_light","dotted_to_striped_mixed4c_light","dotted_to_striped_mixed4d_light","dotted_to_striped_mixed4e_light","dotted_to_striped_mixed5a_light","dotted_to_striped_mixed5b_light"]  #
# concepts = ["dotted","striped"]  #
# concepts = ["dotted","striped","zigzagged"]  # 
# concepts = ["tmp"]  # 

# target = 'honeycomb'  
# concepts =["honeycombed"] 
# concepts =["honeycombed_paisley_mixed3a", "honeycombed_paisley_mixed3b", "honeycombed_paisley_mixed4a", "honeycombed_paisley_mixed4b", "honeycombed_paisley_mixed4c", "honeycombed_paisley_mixed4d", "honeycombed_paisley_mixed4e", "honeycombed_paisley_mixed5a", "honeycombed_paisley_mixed5b"] 
# concepts =["honeycombed","paisley"] 
# concepts =["honeycombed","stratified","paisley"] 

target = 'spider web'  
# concepts =["cobwebbed"] 
# concepts =["cobwebbed_porous_mixed3a", "cobwebbed_porous_mixed3b", "cobwebbed_porous_mixed4a", "cobwebbed_porous_mixed4b", "cobwebbed_porous_mixed4c", "cobwebbed_porous_mixed4d", "cobwebbed_porous_mixed4e", "cobwebbed_porous_mixed5a", "cobwebbed_porous_mixed5b"] 
# concepts =["cobwebbed","porous"] 
concepts =["cobwebbed","porous","blotchy"] 

# target = 'chainlink fence'  #失败
# concepts =["grid","potholed","knitted"]  
#  
# target = 'garter snake'  #失败
# concepts =["scaly","chequered","crystalline"]    # @param

#******************
# # target = 'leopard'  
# concepts =["dotted","scaly","honeycombed"]  
# target = 'tile roof'  
# concepts =["grid","interlaced","stratified","chequered"]   # @param



concepts_string = "_".join(concept.replace(" ", "") for concept in concepts)
is_attack = False
# is_attack = True
attacked_layer_name = "mixed5a"
# fuse_input = "input_cavs"
# fuse_input = "none_fuse"
fuse_input = "aligned_cavs"
# concept_map_type = "reconstructed_cavs"
concept_map_type = "original_cavs"
hidden_dims = [4096]
# embed_dim = 1024
embed_dim = 2048
# embed_dim = 4096
# model_to_run = 'ResNet50V2'
# model_to_run = 'MobileNetV2'
model_to_run = 'GoogleNet'
#******************

# nce_loss = 1
# con_loss = 3
nce_loss = 1
con_loss = 3
k1 =  nce_loss / con_loss

# var_loss = 3
# sim_loss = 1
var_loss = 3
sim_loss = 1
k2 = var_loss / sim_loss

save_dir = "/p/realai/zhenghao/CAVFusion/analysis/"
source_dir = '/p/realai/zhenghao/CAVFusion/data'

dim_align_method="autoencoder"
fuse_method="transformer"

dropout = 0.5

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")




user = 'zhenghao'
# the name of the parent directory that results are stored (only if you want to cache)
project_name = 'tcav_class_test'
working_dir = "/tmp/" + user + '/' + project_name + '/' + model_to_run
# where activations are stored (only if your act_gen_wrapper does so)
sess = utils.create_session()


if not is_attack:
    activation_dir =  working_dir+ '/activations/'
    cav_dir = os.path.join("/p/realai/zhenghao/CAVFusion/analysis/", model_to_run, "original_cavs")
else:
    activation_dir =  working_dir+ '/attacked_activations/'
    cav_dir = os.path.join("/p/realai/zhenghao/CAVFusion/analysis/", model_to_run, "attacked_cavs")

if model_to_run == 'ResNet50V2':
    # bottlenecks = [  'conv4_block4_out', 'conv4_block5_out', 'conv4_block6_out', 'conv5_block1_out', 'conv5_block2_out', 'conv5_block3_out']
    bottlenecks = ['conv4_block1_out', 'conv4_block2_out', 'conv4_block3_out', 'conv4_block4_out', 'conv4_block5_out', 'conv4_block6_out', 'conv5_block1_out', 'conv5_block2_out', 'conv5_block3_out']
    LABEL_PATH = source_dir + "/resnet50_v2/imagenet_labels.txt"
    GRAPH_PATH = source_dir + "/resnet50_v2/resnet50v2_frozen.pb"
    mymodel = model.ResNet50V2Wrapper_public(sess,
                                        GRAPH_PATH,
                                        LABEL_PATH)
elif model_to_run == 'MobileNetV2':
    bottlenecks = ['expanded_conv_2', 'expanded_conv_4', 'expanded_conv_5', 'expanded_conv_7', 'expanded_conv_8', 'expanded_conv_9', 'expanded_conv_11', 'expanded_conv_12', 'expanded_conv_14', 'expanded_conv_15']
    GRAPH_PATH = source_dir + "/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb"
    LABEL_PATH = source_dir + "/mobilenet_v2_1.0_224/mobilenet_v2_label_strings.txt"
    mymodel = model.MobilenetV2Wrapper_public(sess,
                                        GRAPH_PATH,
                                        LABEL_PATH)
elif model_to_run == 'GoogleNet':
    bottlenecks = [ 'mixed3a','mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']
    GRAPH_PATH = source_dir + "/inception5h/tensorflow_inception_graph.pb"
    LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"
    mymodel = model.GoogleNetWrapper_public(sess,
                                            GRAPH_PATH,
                                            LABEL_PATH)