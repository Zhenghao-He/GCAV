import os
import shutil
import random

input_dir = "/p/realai/zhenghao/CAVFusion/data/CUB_200_2011/images"
output_dir = "/p/realai/zhenghao/CAVFusion/data/bird"
os.makedirs(output_dir, exist_ok=True)

# 遍历所有图像路径
image_paths = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(root, file))

# 随机打乱后选取前 500 张
random.seed(42)
random.shuffle(image_paths)
selected = image_paths[:500]

# 拷贝
for i, path in enumerate(selected):
    dst = os.path.join(output_dir, f"bird_{i:03d}.jpg")
    shutil.copy(path, dst)

print("✅ 已成功提取 500 张鸟类图像至：", output_dir)
