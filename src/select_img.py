import json
import os
import shutil

# 路径设定
anno_path = "./data/coco_annotations/annotations/instances_val2017.json"  # COCO标注文件路径
image_dir = "./data/val2017"                              # COCO val 图像路径
output_dir = "./data/car"                                 # 想要保存car图像的目录

# 创建 car 输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载 COCO 标注文件
with open(anno_path, 'r') as f:
    coco = json.load(f)

# 找出“car”类别ID
car_id = next(cat["id"] for cat in coco["categories"] if cat["name"] == "car")

# 所有 car 图像的 image_id
car_image_ids = set(ann["image_id"] for ann in coco["annotations"] if ann["category_id"] == car_id)

# 找到这些 image_id 对应的文件名
car_filenames = [img["file_name"] for img in coco["images"] if img["id"] in car_image_ids]

# 截取前500张
car_filenames = car_filenames[:500]

# 复制图像到目标文件夹
for fname in car_filenames:
    src_path = os.path.join(image_dir, fname)
    dst_path = os.path.join(output_dir, fname)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

print(f"共复制 {len(car_filenames)} 张 car 图像到 `{output_dir}/` 文件夹。")
