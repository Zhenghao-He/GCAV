import os
import urllib.request
from PIL import Image
from io import BytesIO

# 参数设置
url_file = "../data/imagenet.synset.geturls?wnid=n01558993"
save_dir = "../data/robin"
os.makedirs(save_dir, exist_ok=True)

# 加载 URL 列表
with open(url_file, "r") as f:
    urls = [line.strip() for line in f if line.startswith("http")]

print(f"共读取 {len(urls)} 个图像链接，准备下载前 500 张...")

# 下载图像
downloaded = 0
for i, url in enumerate(urls):
    if downloaded >= 500:
        break
    try:
        response = urllib.request.urlopen(url, timeout=5)
        img_data = response.read()
        image = Image.open(BytesIO(img_data))
        image.verify()  # 检查图像是否有效

        filename = os.path.join(save_dir, f"robin_{downloaded}.jpg")
        with open(filename, "wb") as f:
            f.write(img_data)

        print(f"[{downloaded}] 成功下载: {url}")
        downloaded += 1
    except Exception as e:
        print(f"[跳过] 下载失败 ({i}): {url} —— {e}")
