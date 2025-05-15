import tcav.activation_generator as act_gen
# # import tcav.cav as cav
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils
import tcav.utils_plot as utils_plot # utils_plot requires matplotlib
import os 
# import tensorflow as tf
import absl
from get_cavs import get_cavs

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

import tensorflow as tf
from align_dim import CAVAutoencoder
from configs import alphas, concepts, bottlenecks, target, save_dir, dim_align_method, fuse_method, model_to_run, embed_dim, hidden_dims, dropout, device, num_random_exp, concepts_string, fuse_input, source_dir, working_dir, activation_dir, mymodel, save_dir,sess, is_attack

from configs import k1, k2, embed_dim

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# tf.config.run_functions_eagerly(True)

def plot_results(results, random_counterpart=None, random_concepts=None, num_random_exp=100, min_p_val=0.05, save_path="/p/realai/zhenghao/CAVFusion/analysis/"):
    # 打开日志文件
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, f"log_{'attacked' if is_attack else 'original'}_{concepts_string}.txt")
    # log_path = os.path.join(save_path, f"log_{'attacked' if is_attack else 'original'}_{concepts_string}.txt")
    with open(log_path, "w") as log:
        # Helper function: 判断是否是随机概念
        def is_random_concept(concept):
            if random_counterpart:
                return random_counterpart == concept
            elif random_concepts:
                return concept in random_concepts
            else:
                return 'random500_' in concept

        # 打印类别信息
        print("Class =", results[0]['target_class'], file=log)

        # 数据准备
        result_summary = {}  # 按概念和瓶颈整理结果
        random_i_ups = {}  # 存储随机结果的 i_up

        for result in results:
            if result['cav_concept'] not in result_summary:
                result_summary[result['cav_concept']] = {}
            if result['bottleneck'] not in result_summary[result['cav_concept']]:
                result_summary[result['cav_concept']][result['bottleneck']] = []
            result_summary[result['cav_concept']][result['bottleneck']].append(result)

            # 存储随机结果
            if is_random_concept(result['cav_concept']):
                if result['bottleneck'] not in random_i_ups:
                    random_i_ups[result['bottleneck']] = []
                random_i_ups[result['bottleneck']].append(result['i_up'])

        # 准备绘图数据
        plot_data = {}
        plot_concepts = []

        for concept in result_summary:
            if not is_random_concept(concept):
                print(" ", "Concept =", concept, file=log)
                plot_concepts.append(concept)

                for bottleneck in result_summary[concept]:
                    i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]

                    # 计算统计显著性
                    _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)

                    if bottleneck not in plot_data:
                        plot_data[bottleneck] = {'bn_vals': [], 'bn_stds': [], 'significant': []}

                    if p_val > min_p_val:
                        # 不显著，设置为默认值
                        plot_data[bottleneck]['bn_vals'].append(0.01)
                        plot_data[bottleneck]['bn_stds'].append(0)
                        plot_data[bottleneck]['significant'].append(False)
                    else:
                        # 统计显著
                        plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
                        plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))
                        plot_data[bottleneck]['significant'].append(True)

                    print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
                          "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
                              bottleneck, np.mean(i_ups), np.std(i_ups),
                              np.mean(random_i_ups[bottleneck]),
                              np.std(random_i_ups[bottleneck]), p_val,
                              "not significant" if p_val > min_p_val else "significant"),
                          file=log)

        # 确定非随机概念的数量
        if random_counterpart:
            num_concepts = len(result_summary) - 1
        elif random_concepts:
            num_concepts = len(result_summary) - len(random_concepts)
        else:
            num_concepts = len(result_summary) - num_random_exp

        num_bottlenecks = len(plot_data)
        bar_width = 0.35
        index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)

        # 绘制图表
        fig, ax = plt.subplots()
        for i, (bn, vals) in enumerate(plot_data.items()):
            bar = ax.bar(index + i * bar_width, vals['bn_vals'],
                         bar_width, yerr=vals['bn_stds'], label=bn)

            # 标注统计不显著的柱子
            for j, significant in enumerate(vals['significant']):
                if not significant:
                    ax.text(index[j] + i * bar_width - 0.1, 0.01, "*",
                            fontdict={'weight': 'bold', 'size': 16,
                                      'color': bar.patches[0].get_facecolor()})
        
        # 打印绘图数据到日志
        print("Plot Data =", plot_data, file=log)

        # 设置图表属性
        ax.set_title('TCAV Scores for each concept and bottleneck')
        ax.set_ylabel('TCAV Score')
        ax.set_xticks(index + num_bottlenecks * bar_width / 2)
        ax.set_xticklabels(plot_concepts)
        ax.legend()

        # 保存图表
        fig.tight_layout()
        # pic_path = os.path.join(save_path, f"tcav_results_{'attacked' if is_attack else 'original'}_{concepts_string}.png")
        # plt.savefig(pic_path)
        # print(f"Plot saved to {pic_path}")

# This is the name of your model wrapper (InceptionV3 and GoogleNet are provided in model.py)

# user = 'zhenghao'
# # the name of the parent directory that results are stored (only if you want to cache)
# project_name = 'tcav_class_test'
# working_dir = "/tmp/" + user + '/' + project_name
# # where activations are stored (only if your act_gen_wrapper does so)
# activation_dir =  working_dir+ '/activations/'
# # where CAVs are stored. 
# # You can say None if you don't wish to store any.



original_cavs_path = os.path.join(save_dir, model_to_run, "original_cavs")

cavs = np.load(os.path.join(original_cavs_path,f"cavs_{concepts[0]}.npy"), allow_pickle=True)


autoencoders = CAVAutoencoder(input_dims=[len(cav[0]) for cav in cavs], embed_dim=embed_dim,hidden_dims=hidden_dims, dropout=dropout , device=device, save_dir=os.path.join(save_dir,model_to_run))

cav_dir = os.path.join(save_dir, model_to_run, "reconstructed_cavs", dim_align_method, fuse_method, autoencoders.key_params, fuse_input, f"{k1:.2f}_{k2:.2f}")
# cav_dir = os.path.join(save_dir, model_to_run, "reconstructed_cavs", dim_align_method, fuse_method, "[4096,4096]_0.5", fuse_input)
# # where the images live.
# save_path = save_dir
# # TODO: replace 'YOUR_PATH' with path to downloaded models and images. 
# source_dir = '/p/realai/zhenghao/CAVFusion/data'

# bottlenecks = [ 'mixed3a']  # @param 
if __name__ == "__main__":      
    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(cav_dir)


    # # Create TensorFlow session.
    # sess = utils.create_session()

    # # GRAPH_PATH is where the trained model is stored.
    # GRAPH_PATH = source_dir + "/inception5h/tensorflow_inception_graph.pb"
    # LABEL_PATH is where the labels are stored. Each line contains one class, and they are ordered with respect to their index in 
    # the logit layer. (yes, id_to_label function in the model wrapper reads from this file.)
    # For example, imagenet_comp_graph_label_strings.txt looks like:
    # dummy                                                                                      
    # kit fox
    # English setter
    # Siberian husky ...

    # LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"

    # mymodel = model.GoogleNetWrapper_public(sess,
    #                                         GRAPH_PATH,
    #                                         LABEL_PATH)

    act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)


    absl.logging.set_verbosity(0)
    ## only running num_random_exp = 10 to save some time. The paper number are reported for 500 random runs. 
    mytcav = tcav.TCAV(sess,
                    target,
                    concepts,
                    bottlenecks,
                    act_generator,
                    alphas,
                    cav_dir=cav_dir,
                    num_random_exp=num_random_exp)#10)

    print('running TCAV')
    results = mytcav.run(run_parallel=False)
    print('done!')
    utils_plot.plot_results=plot_results # plot to file
    save_path = os.path.join(save_dir, model_to_run, "recostructed_results", fuse_input, f"{k1:.2f}_{k2:.2f}_{embed_dim}")
    os.makedirs(save_path, exist_ok=True)
    utils_plot.plot_results(results, num_random_exp=num_random_exp,save_path=save_path)