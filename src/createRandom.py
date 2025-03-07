import os
import random
import shutil
from PIL import Image  # 用于图片格式转换

def distribute_images(src_folder, dst_folder, num_folders=1, images_per_folder=50):
    """
    随机将文件夹中的 JPEG 图片分配到多个子文件夹中，并统一转换为 .jpg 格式（文件将被复制）。
    
    :param src_folder: 原始图片文件夹路径
    :param dst_folder: 目标文件夹路径
    :param num_folders: 子文件夹数量
    :param images_per_folder: 每个子文件夹的图片数量
    """
    # 获取文件夹中的所有 JPEG 图片文件（包括大小写扩展名）
    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpeg', '.jpg'))]
    if len(images) < num_folders * images_per_folder:
        print(f"图片数量不足，至少需要 {num_folders * images_per_folder} 张图片！")
        return
    
    # 打乱图片列表
    random.shuffle(images)
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(dst_folder, exist_ok=True)
    
    # 开始分配图片
    for i in range(num_folders):
        folder_name = f"random500_{i}"  # 子文件夹名称
        folder_path = os.path.join(dst_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # 获取当前子文件夹的图片
        start_index = i * images_per_folder
        end_index = start_index + images_per_folder
        current_images = images[start_index:end_index]
        
        # 处理图片：转换为 .jpg 并复制到目标文件夹
        for img in current_images:
            src_path = os.path.join(src_folder, img)
            dst_image_name = os.path.splitext(img)[0] + ".jpg"
            dst_path = os.path.join(folder_path, dst_image_name)
            
            try:
                # 使用 PIL 转换为 .jpg 格式并保存到目标文件夹
                with Image.open(src_path) as image:
                    image = image.convert("RGB")  # 确保转换为 RGB 模式
                    image.save(dst_path, "JPEG")
            except Exception as e:
                print(f"无法处理图片 {src_path}: {e}")
    
    print(f"图片已成功分配到 {num_folders} 个子文件夹，并转换为 .jpg 格式。")



src_folder = "/p/realai/zhenghao/CAVFusion/data/test"  # 原始图片文件夹路径
dst_folder = "/p/realai/zhenghao/CAVFusion/data"  # 目标文件夹路径
distribute_images(src_folder, dst_folder)
