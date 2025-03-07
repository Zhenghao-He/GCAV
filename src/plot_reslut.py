import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm
from configs import model_to_run, save_dir, concepts_string, bottlenecks,target

TITLE_FONT_SIZE = 26    # 标题字体大小
LABEL_FONT_SIZE = 22    # 标签字体大小
TICK_FONT_SIZE = 20     # 刻度字体大小

import re

def parse_tcav_file(filename):
    """
    解析 TCAV 结果文件，提取每个概念在不同层的 TCAV 分数、标准差和显著性信息。

    Parameters:
        filename (str): TCAV 结果文件路径

    Returns:
        concept_means (dict): {concept: {layer: mean_tcav}}
        concept_stds (dict): {concept: {layer: std_tcav}}
        concept_significance (dict): {concept: {layer: significant (bool)}}
    """
    concept_means = {}       # 存储 TCAV 均值
    concept_stds = {}        # 存储 TCAV 标准差
    concept_significance = {}  # 存储显著性信息
    current_concept = None

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Concept ="):
                # 解析 Concept
                parts = line.split("=")
                if len(parts) >= 2:
                    current_concept = parts[1].strip()
                    if current_concept not in concept_means:
                        concept_means[current_concept] = []
                        concept_stds[current_concept] = []
                        concept_significance[current_concept] = []

            elif "Bottleneck =" in line and "TCAV Score =" in line:
                # 解析 Bottleneck 层、TCAV 分数和标准差
                m = re.search(r"Bottleneck = ([^ ]+)\. TCAV Score = ([0-9.]+) \(\+\- ([0-9.]+)\),.*p-val = ([0-9.]+) \((significant|not significant)\)", line)
                if m and current_concept is not None:
                    layer = m.group(1)        # 层的名称
                    score_mean = float(m.group(2))  # TCAV 均值
                    score_std = float(m.group(3))   # TCAV 标准差
                    p_value = float(m.group(4))     # p 值
                    significant = (m.group(5) == "significant")  # 是否显著

                    # 存入对应的字典
                    concept_means[current_concept].append(score_mean)
                    concept_stds[current_concept].append(score_std)
                    concept_significance[current_concept].append(significant)

    return concept_means, concept_stds, concept_significance


import numpy as np
from scipy.stats import skew, iqr

def compute_statistics(concept_scores):
    stats = {}
    for concept, scores in concept_scores.items():
        arr = np.array(scores)
        mean = np.mean(arr)
        variance = np.var(arr)
        std_dev = np.std(arr)  # 标准差
        median = np.median(arr)  # 中位数
        min_val = np.min(arr)  # 最小值
        max_val = np.max(arr)  # 最大值
        skewness = skew(arr)  # 偏度
        cv = std_dev / mean if mean != 0 else 0  # 变异系数（Coefficient of Variation）
        iqr_val = iqr(arr)  # 四分位距（Interquartile Range）
        range_ratio = (max_val - min_val) / mean if mean != 0 else 0  # 最大最小比

        stats[concept] = {
            "mean": mean,
            "variance": variance,
            "std_dev": std_dev,
            "median": median,
            "min": min_val,
            "max": max_val,
            "skewness": skewness,
            "cv": cv,  # 变异系数
            "iqr": iqr_val,  # 四分位距
            "range_ratio": range_ratio  # 最大最小比
        }
    return stats

def save_statistics(stats, output_filename):
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("Concept\tMean\tVariance\tStdDev\tMedian\tMin\tMax\tSkewness\tCV\tIQR\tRangeRatio\n")
        for concept, stat in stats.items():
            f.write(f"{concept}: & {stat['mean']:.3f} & {stat['std_dev']:.3f} & {stat['cv']:.3f} & {stat['range_ratio']:.3f}\n")

            f.write(f"{concept}\t{stat['mean']:.3f}\t{stat['variance']:.3f}\t"
                    f"{stat['std_dev']:.3f}\t{stat['median']:.3f}\t"
                    f"{stat['min']:.3f}\t{stat['max']:.3f}\t"
                    f"{stat['skewness']:.3f}\t{stat['cv']:.3f}\t"
                    f"{stat['iqr']:.3f}\t{stat['range_ratio']:.3f}\n")


def plot_violin(concept_scores, output_filename, type):

    data = []
    for concept, scores in concept_scores.items():
        for s in scores:
            data.append((concept, s))
    df = pd.DataFrame(data, columns=["Concept", "TCAV_Score"])
    
    # 设置调色盘，确保每个 concept 颜色不同
    unique_concepts = df["Concept"].unique()
    palette = sns.color_palette("Dark2_r", len(unique_concepts))  # tab10 颜色亮丽，可自定义

    plt.figure(figsize=(10, 6))

    sns.violinplot(x="Concept", y="TCAV_Score", data=df, palette=palette)

    if type == "original":
        plt.title(f"$\\bf{{{target}}}$ - original method on {model_to_run}",
            fontsize=TITLE_FONT_SIZE)
    elif type == "proposed":
        plt.title(f"$\\bf{{{target}}}$ - proposed method on {model_to_run}",
                fontsize=TITLE_FONT_SIZE)



    # plt.title(f"{type} Violin Plot of TCAV Scores per Concept - {model_to_run}", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Concept", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("TCAV Score", fontsize=LABEL_FONT_SIZE)

    plt.xticks(fontsize=TICK_FONT_SIZE)  # 调整x轴刻度字体大小
    plt.yticks(fontsize=TICK_FONT_SIZE)  # 调整y轴刻度字体大小
    # 添加网格线以提高可读性
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
def plot_tcav_bar_chart(concept_scores,layer_names,concept_stds,output_filename,type):
    """
    Plots a bar chart of TCAV scores for different concepts.
    
    Parameters:
        concept_scores (dict): A dictionary where each key is a concept name and 
                               each value is a list of TCAV scores.
    """
    # Extract concepts and compute statistics
    num_layers = len(layer_names)
    num_concepts = len(concept_scores)
    
    x = np.arange(num_concepts)  # x 轴上的概念位置
    width = 0.1  # 每个柱形的宽度

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))  # 生成不同层的颜色

    for i, layer in enumerate(layer_names):
        means = [concept_scores[concept][i] for concept in concept_scores]  # 提取均值
        # std_devs = [concept_stds[concept][i] for concept in concept_stds]  # 提取标准差

        ax.bar(x + i * width, means, width, label=layer, color=colors[i], 
               alpha=0.8, capsize=4, error_kw={'elinewidth': 1.5})  # 误差棒

    ax.set_xticks(x + (num_layers / 2) * width)  # 调整 x 轴标签位置
    # ax.set_xticklabels(concept_scores.keys(), ha="right")
    ax.set_xticklabels(list(concept_scores.keys()))

    # ax.set_xlabel("Concepts", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("TCAV Score", fontsize=LABEL_FONT_SIZE)


    if type == "original":
        ax.set_title(rf"$\bf{{{target}}}$ - original method on {model_to_run}",
                    fontsize=TITLE_FONT_SIZE)
    elif type == "proposed":
        ax.set_title(rf"$\bf{{{target}}}$ - proposed method on {model_to_run}",
                    fontsize=TITLE_FONT_SIZE)


    # ax.set_title(f"{type} TCAV Scores with Variance Across Layers - {model_to_run}", fontsize=TITLE_FONT_SIZE)
    ax.legend(title="Layer", bbox_to_anchor=(0.95, 0.95), loc="upper left")
    # ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95), fontsize=12)
    ax.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def plot_bell_curves(concept_scores, output_filename, type):

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors 
    for idx, (concept, scores) in enumerate(concept_scores.items()):
        arr = np.array(scores)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            std = 0.01
        x_vals = np.linspace(mean - 3*std, mean + 3*std, 100)
        y_vals = norm.pdf(x_vals, mean, std)
        plt.plot(x_vals, y_vals, 
                 label=f"{concept} (mean={mean:.2f}, std={std:.2f})",
                 color=colors[idx % len(colors)])
    plt.title(f"{type} Bell Curves (Normal Distributions) of TCAV Scores per Concept - {model_to_run}", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("TCAV Score", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Probability Density", fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)  # 调整x轴刻度字体大小
    plt.yticks(fontsize=TICK_FONT_SIZE)  # 调整y轴刻度字体大小
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def process_tcav_file(save_path, input_filename, stats_output_filename, violin_output_filename, bell_output_filename,bar_output_filename, type,layer_names):
    input_filename = os.path.join(save_path, input_filename)
    stats_output_filename = os.path.join(save_path, stats_output_filename)
    violin_output_filename = os.path.join(save_path, violin_output_filename)
    bell_output_filename = os.path.join(save_path, bell_output_filename)
    bar_output_filename = os.path.join(save_path, bar_output_filename)

    concept_scores, concept_stds,_ = parse_tcav_file(input_filename)

    stats = compute_statistics(concept_scores)
    
 
    save_statistics(stats, stats_output_filename)
    
 
    plot_violin(concept_scores, violin_output_filename, type)
    
  
    plot_bell_curves(concept_scores, bell_output_filename, type)
    
    plot_tcav_bar_chart(concept_scores,layer_names,concept_stds, bar_output_filename, type)
    print("处理完成，统计结果和图片均已保存。")


if __name__ == "__main__":
    # type = "original"
    type = "proposed"
    if type == "original":
        save_path = os.path.join(save_dir, model_to_run, "original_results")
    elif type == "proposed":
        save_path = os.path.join(save_dir, model_to_run, "recostructed_results")
 
    # input_filename = f"log_original_{concepts_string}.txt"             
    input_filename = "log_attacked_DottedToStriped.txt"             
    # input_filename = f"log_{concepts_string}.txt"             
    stats_output_filename = f"tcav_stats_{concepts_string}.txt"   
    violin_output_filename = f"tcav_violin_{concepts_string}.png" 
    bell_output_filename = f"tcav_bell_{concepts_string}.png"    
    bar_output_filename = f"tcav_bar_{concepts_string}.png"
    
    process_tcav_file(save_path, input_filename, stats_output_filename, violin_output_filename, bell_output_filename,bar_output_filename, type,bottlenecks)
