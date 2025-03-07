import os
import cv2
import numpy as np
from configs import save_dir, model_to_run, concept_map_type, target, concepts
# 设定根目录路径
root_dir = os.path.join(save_dir, model_to_run, "concept_maps", concept_map_type, target, concepts[0])
if not os.path.exists(root_dir):
    raise ValueError(f"Concept maps not found at {root_dir}")
# 获取所有子文件夹，并按顺序排序
subfolders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])

# 获取所有图片的文件名（假设所有子文件夹中图片名字是相同的）
image_files = sorted(os.listdir(os.path.join(root_dir, subfolders[0])))

# 确保输出文件夹存在
output_dir = os.path.join(root_dir, "stitched_images")
os.makedirs(output_dir, exist_ok=True)

# 遍历每个图片名称
for image_name in image_files:
    image_list = []

    # 读取每个子文件夹中相同名称的图片
    for subfolder in subfolders:
        image_path = os.path.join(root_dir, subfolder, image_name)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                image_list.append(img)

    # 进行拼接（按垂直方向拼接，也可以改成水平拼接）
    if image_list:
        stitched_image = np.hstack(image_list)  # 改成 np.hstack(image_list) 可水平拼接
        cv2.imwrite(os.path.join(output_dir, image_name), stitched_image)

print(f"拼接完成，图片保存在: {output_dir}")
