import os
import torch
import cv2
import numpy as np
from image_quality_assessment import PSNR, SSIM
from utils import Summary, AverageMeter

# 假设这两个模型已经定义并且加载好了
psnr_model = PSNR(crop_border=0, only_test_y_channel=False, data_range=1.0)
ssim_model = SSIM(crop_border=0, only_test_y_channel=False, data_range=255.0)

# psnr_model = psnr_model.to(device)
# ssim_model = ssim_model.to(device)


def calculate_metrics(original, reconstructed):
    # 将图像转换为Tensor并归一化
    original_tensor = torch.tensor(original).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    reconstructed_tensor = torch.tensor(reconstructed).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 计算PSNR和SSIM
    psnr = psnr_model(reconstructed_tensor, original_tensor)
    ssim = ssim_model(reconstructed_tensor, original_tensor)

    return psnr.item(), ssim.item()


def main(data_folder, data1_folder):
    # 创建AverageMeter实例
    psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
    ssimes = AverageMeter("SSIM", ":4.4f", Summary.AVERAGE)

    # 获取所有重建图像的文件名
    reconstructed_images = os.listdir(data_folder)

    # 遍历每一张重建图像
    for image_name in reconstructed_images:
        if image_name.endswith('.png'):
            # 构建完整路径
            reconstructed_path = os.path.join(data_folder, image_name)
            original_path = os.path.join(data1_folder, image_name)

            # 读取图像
            reconstructed_image = cv2.imread(reconstructed_path)
            original_image = cv2.imread(original_path)

            # 确保两张图像都被正确读取
            if reconstructed_image is not None and original_image is not None:
                psnr_value, ssim_value = calculate_metrics(original_image, reconstructed_image)

                # 更新平均值
                psnres.update(psnr_value, reconstructed_image.shape[0])
                ssimes.update(ssim_value, reconstructed_image.shape[0])

                print(f"Image: {image_name} | PSNR: {psnr_value:.2f} | SSIM: {ssim_value:.4f}")
            else:
                print(f"Error reading images for {image_name}")

    # 输出最终的平均PSNR和SSIM
    print(f"Average PSNR: {psnres.avg:.2f}, Average SSIM: {ssimes.avg:.4f}")


if __name__ == "__main__":
    main(r'D:\12045\DeepFool-master\deepfool_attack_test', r'D:\12045\DeepFool-master\data100_resize')
