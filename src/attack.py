import torch
import torch.nn as nn
import torch.optim as optim
from configs import embed_dim, dim_align_method, fuse_method, concepts, bottlenecks, hidden_dims, dropout,save_dir, model_to_run, num_random_exp, concepts_string, fuse_input,target, concept_map_type, attacked_layer_name, save_dir
import tcav.utils as utils
import tcav.model  as model
import cv2
import tcav.activation_generator as act_gen
import numpy as np
import tensorflow as tf
import os
from configs import GRAPH_PATH, LABEL_PATH, mymodel, source_dir, activation_dir

def gen_image_from_tensor(perturbed_images, original_images, save_dir, attacked_layer_name):
    
    perturbed_save_dir = os.path.join(save_dir, attacked_layer_name)
    os.makedirs(perturbed_save_dir, exist_ok=True)
    original_save_dir = os.path.join(save_dir, "original_images")
    os.makedirs(original_save_dir, exist_ok=True)
    combined_save_dir = os.path.join(save_dir, "combined_images")
    os.makedirs(combined_save_dir, exist_ok=True)
    for img_idx, (perturbed_image, original_image) in enumerate(zip(perturbed_images, original_images)):
        perturbed_image = (perturbed_image * 255).astype(np.uint8)
        original_image = (original_image * 255).astype(np.uint8)
        # 保存原图
        cv2.imwrite(os.path.join(original_save_dir, f"original_image_{img_idx}.png"), original_image)
        # 保存扰动后的图
        cv2.imwrite(os.path.join(perturbed_save_dir, f"perturbed_image_{img_idx}.png"), perturbed_image)
          # 如果图片尺寸不一致，可以先进行resize调整为相同尺寸
        if perturbed_image.shape != original_image.shape:
            original_image = cv2.resize(original_image, (perturbed_image.shape[1], perturbed_image.shape[0]))
        
        # 横向拼接两张图
        combined_image = cv2.hconcat([original_image, perturbed_image])
        # 或者使用 numpy.hstack: combined_image = np.hstack((original_image, perturbed_image))
        
        # 保存拼接后的图片
        cv2.imwrite(os.path.join(combined_save_dir, f"combined_image_{img_idx}.png"), combined_image)


def pgd_attack_on_pb(pb_file, input_images, attacked_layer_name, target_mu,
                     input_tensor_name='input:0', input_shape=None,
                     epsilon=8.0/255.0, step_size=2.0/255.0, num_steps=20):
    """
    Perform a PGD attack on a TensorFlow pb model targeting a specific layer.
    """
    if input_shape is None:
        input_shape = [None] + list(input_images.shape[1:])

    with tf.Graph().as_default() as graph:
        x = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name="input_images")
        delta = tf.Variable(tf.zeros_like(x), trainable=True, name="delta")

        perturbed_input = tf.clip_by_value(x + delta, 0, 1)  # 确保扰动后图像在 [0,1] 范围

        with tf.io.gfile.GFile(pb_file, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, input_map={input_tensor_name: perturbed_input}, name="")

        features = graph.get_tensor_by_name(attacked_layer_name)
        if not isinstance(target_mu, tf.Tensor):
            target_mu = tf.constant(target_mu, dtype=tf.float32)

        # print("features shape:", features.shape)
        # print("target_mu shape:", target_mu.shape)

        features = tf.nn.l2_normalize(features, axis=-1)
        target_mu = tf.nn.l2_normalize(target_mu, axis=-1)

        # cos_sim = tf.reduce_sum(features * target_mu, axis=-1) / (tf.norm(features, axis=-1) * tf.norm(target_mu, axis=-1) + 1e-8)
        # loss = 1-tf.reduce_mean(cos_sim)  # maximize cosine similarity
        loss = tf.reduce_mean(tf.square(features - target_mu))
        grad_delta = tf.gradients(loss, delta)[0]

        new_delta = tf.clip_by_value(delta - step_size * tf.sign(grad_delta), -epsilon, epsilon)
        update_op = tf.compat.v1.assign(delta, new_delta)

        final_perturbed = tf.clip_by_value(x + delta, 0, 1)  # 确保像素值在 [0,1] 之间

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(delta.initializer)

        for i in range(num_steps):
            loss_val, _ = sess.run([loss, update_op], feed_dict={x: input_images})
            print("Step %d, loss: %f" % (i, loss_val))

        perturbed_images = sess.run(final_perturbed, feed_dict={x: input_images})

    return perturbed_images


# Example usage:
if __name__ == '__main__':
    # Simulated example:
    # Assume we have a batch of concept images (e.g., tokens) with shape (N, C, H, W)
    # batch_size = 8
    # C, H, W = 3, 32, 32  # For example, CIFAR-like images
    # feature_dim = 128   # Dimension of the hidden features
    # x = torch.rand(batch_size, C, H, W)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    # import pdb; pdb.set_trace()
    assert len(concepts) == 2, "Only support 2 concepts attack"
    ori_concept = concepts[0] # concept be attacked
    tar_concept = concepts[1] # concept to be aligned
    save_dir_whole = os.path.join(save_dir, model_to_run, "attack", f"{ori_concept}_{tar_concept}")
    for bottleneck in bottlenecks:
        attacked_layer_name = bottleneck
        print(f"Attacking layer {attacked_layer_name}")

        act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)
        class_examples = act_generator.get_examples_for_concept(ori_concept)
        save_dir = os.path.join(save_dir_whole, attacked_layer_name)

        

        class_acts_layer = {} # (layer_n, num_exs, acts)
        acts = act_generator.process_and_load_activations(bottlenecks, concepts)
        # for bottleneck in bottlenecks:
        acts_instance = acts[tar_concept][attacked_layer_name]# (num_exs, H, W, D)
        # target_mu = np.mean(acts_instance, axis=0)
        target_mu = np.mean(acts_instance, axis=0, keepdims=True)  # 这样 target_mu 形状变成 [1, H, W, D]

        # target_mu = acts_instance

        # import pdb; pdb.set_trace()


            # Replace these with your actual data and parameters
        pb_model_path = GRAPH_PATH
        # Assume input_images is a numpy array of shape [N, H, W, C]
        
        # Specify the attacked layer's tensor name (adjust according to your model)
        
        # target_mu should be computed from your target concept images; here we use random values for demonstration
        # target_mu = np.random.randn(2048).astype(np.float32)


        perturbed_images = pgd_attack_on_pb(pb_file=GRAPH_PATH, #pb_file, input_images, attacked_layer_name, target_mu
                                        input_images=class_examples, 
                                        attacked_layer_name=attacked_layer_name+":0", 
                                        target_mu=target_mu,
                                        input_tensor_name='input:0',
                                        input_shape=class_examples.shape,
                                        epsilon=8/255,
                                        step_size=2/255,
                                        num_steps=20)

        # Verify maximum perturbation magnitude
        print("Max perturbation:", np.max(np.abs(perturbed_images - class_examples)))

        # for img in class_examples:
        #     img = np.expand_dims(img, axis=0)
        #     # import pdb; pdb.set_trace()
        #     perturbed_images = pgd_attack_on_pb(pb_file=GRAPH_PATH, #pb_file, input_images, attacked_layer_name, target_mu
        #                                         input_images=img, 
        #                                         attacked_layer_name=attacked_layer_name+":0", 
        #                                         target_mu=target_mu,
        #                                         input_tensor_name='input:0',
        #                                         input_shape=img.shape,
        #                                         epsilon=8.0/255.0,
        #                                         step_size=2.0/255.0,
        #                                         num_steps=2000)
            
        #     # Verify maximum perturbation magnitude
        #     print("Max perturbation:", np.max(np.abs(perturbed_images - class_examples)))
            # import pdb; pdb.set_trace()
        gen_image_from_tensor(perturbed_images=perturbed_images, original_images=class_examples, save_dir=save_dir, attacked_layer_name=f"{ori_concept}_{tar_concept}_{attacked_layer_name}")
        # model_f = model_f.to(device)
        # target_mu = target_mu.to(device)
        
        # # Run the Token Pushing attack
        # perturbed_x = pgd_token_push_attack(x, model_f, target_mu,
        #                                       epsilon=8/255, step_size=2/255, num_steps=20)
        
        # # For verification: print maximum change (should be <= epsilon)
        # delta = perturbed_x - x
        # print("Max perturbation:", delta.abs().max().item())
        
        # Optionally, you can visualize x and perturbed_x to confirm that they are almost identical.
