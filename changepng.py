import os
import cv2

def convert_images_to_png(source_dir, target_dir):
    # 创建目标目录，如果不存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 构建完整的文件路径
            file_path = os.path.join(source_dir, filename)
            # 读取图像
            image = cv2.imread(file_path)
            # 构建目标文件的路径
            target_file = os.path.join(target_dir, os.path.splitext(filename)[0] + '.png')
            # 将图像保存为 PNG 格式
            print(f"Saving {target_file}")
            cv2.imwrite(target_file, image)

# 使用函数
source_directory = 'data/my_attack_densenet_350'  # 源目录
target_directory = 'data/my_attack_densenet_350_png'  # 目标目录

convert_images_to_png(source_directory, target_directory)