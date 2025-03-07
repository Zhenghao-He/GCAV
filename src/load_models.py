# from tensorflow.keras.applications import ResNet50
# model = ResNet50(weights='imagenet')
# model.save("resnet50_model.h5")  # 保存为 H5 格式
# import tensorflow as tf

# model = tf.keras.models.load_model("resnet50_model.h5")
# import os
# # os.mkdir("/p/realai/zhenghao/CAVFusion/data/resnet", exist_ok=True)
# os.makedirs("/p/realai/zhenghao/CAVFusion/data/resnet", exist_ok=True)
# tf.saved_model.save(model, "/p/realai/zhenghao/CAVFusion/data/resnet/resnet50_saved_model")
# import tensorflow as tf
import os
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# # 加载预训练模型（基于 ImageNet 的权重）
# model = MobileNetV2(weights='imagenet')

# path = "/p/realai/zhenghao/CAVFusion/data/mobilenet_v2"
# os.makedirs(path, exist_ok=True)
# # 将模型保存为 SavedModel 格式（保存后会生成 saved_model.pb 文件）
# tf.saved_model.save(model, os.path.join(path, "mobilenet_v2_saved_model"))
# # import 
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# 加载 ResNet-50 V2 预训练模型
model = tf.keras.applications.ResNet50V2(weights="imagenet")

# 创建 TensorFlow 函数
@tf.function
def serving_fn(inputs):
    return model(inputs)

concrete_function = serving_fn.get_concrete_function(tf.TensorSpec([None, 224, 224, 3], tf.float32))

# 获取冻结图
frozen_func = convert_variables_to_constants_v2(concrete_function)
import pdb; pdb.set_trace()
path = "/p/realai/zhenghao/CAVFusion/data/resnet50_v2"
os.makedirs(path, exist_ok=True)
# 保存冻结图
tf.io.write_graph(frozen_func.graph, ".", os.path.join(path, "resnet50v2_frozen.pb"), as_text=False)


# from tensorflow.keras.applications.resnet50 import decode_predictions
# import numpy as np

# # 获取 ImageNet 类别标签
# labels = decode_predictions(np.expand_dims(np.arange(1000), axis=0), top=1000)[0]
# imagenet_labels = [label[1] for label in labels]

# # 打印前10个类别
# print(imagenet_labels[:10])
# import os

# # 指定保存路径
# path = "/p/realai/zhenghao/CAVFusion/data/resnet50_v2"
# os.makedirs(path, exist_ok=True)

# # 保存到 TXT 文件，每行一个类别
# with open(os.path.join(path, "imagenet_labels.txt"), "w") as f:
#     f.write("\n".join(imagenet_labels))

# print(f"标签已保存至 {os.path.join(path, 'imagenet_labels.txt')}")

# from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
# # from tensorflow.keras.applications.resnet_v2 import preprocess_input

# import numpy as np
# import tensorflow as tf
# from configs import source_dir, activation_dir, mymodel
# import tcav.activation_generator as act_gen
# from tensorflow.keras.preprocessing import image
# # 载入 ResNet-50 V2 预训练模型
# model = tf.keras.applications.ResNet50V2(weights="imagenet")
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# 加载预训练模型（基于 ImageNet 的权重）
# model = MobileNetV2(weights='imagenet')
# 创建随机输入张量 (模拟 224x224 RGB 图片)
# input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
# act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)
# class_examples = act_generator.get_examples_for_concept("zebra")
# img_path = '/p/realai/zhenghao/CAVFusion/data/zebra/800px-N2_zebra.jpg'  # 替换为实际的图像路径
# img = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img)

# # 预处理输入
# img_array = np.expand_dims(img_array, axis=0)
# input_data = preprocess_input(img_array)
# print(f"Original pixel range: {img_array.min()} - {img_array.max()}")  # 一般应在 0-255
# print(f"Processed pixel range: {input_data.min()} - {input_data.max()}")  # 应该在 -1 到 1

# for layer in model.layers:
#     print(layer.name, layer.input, layer.output)


# # 进行推理
# preds = model.predict(input_data)
# import pdb; pdb.set_trace()
# # 解码前 5 个预测结果
# decoded_preds = decode_predictions(preds, top=5)[0]

# # 打印结果
# for i, (imagenet_id, label, score) in enumerate(decoded_preds):
#     print(f"{i+1}: {label} ({score:.2f})")

