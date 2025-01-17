# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os

import cv2
import torch
from torch import nn

import model
from imgproc import preprocess_one_image, tensor_to_image
from utils import load_pretrained_state_dict
# 生成超分辨率图像的脚本, 使用了一个预训练的深度学习模型来将输入的低分辨率图像转换为高分辨率图像

def main(args):
    device = torch.device(args.device)

    # Read original image
    input_tensor = preprocess_one_image(args.inputs, False, args.half, device)

    # Initialize the model
    sr_model = build_model(args.model_arch_name, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    sr_model = load_pretrained_state_dict(sr_model, args.compile_state, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    sr_model.eval()

    # Enable half-precision inference to reduce memory usage and inference time
    if args.half:
        sr_model.half()

    # Use the model to generate super-resolved images
    with torch.no_grad():
        # Reasoning
        sr_tensor = sr_model(input_tensor)

    # Save image
    cr_image = tensor_to_image(sr_tensor, False, args.half)
    cr_image = cv2.cvtColor(cr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, cr_image)

    print(f"SR image save to `{args.output}`")


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    sr_model = model.__dict__[model_arch_name](in_channels=3,
                                               out_channels=3,
                                               channels=64,
                                               growth_channels=32,
                                               num_rrdb=23)

    sr_model = sr_model.to(device)

    return sr_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",
                        type=str,
                        default="./figure/baboon.png",
                        help="Original image path.")
    parser.add_argument("--output",
                        type=str,
                        default="./figure/ESRGAN_x4_baboon.png",
                        help="Super-resolution image path.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="rrdbnet_x4",
                        help="Model architecture name.")
    parser.add_argument("--compile_state",
                        type=bool,
                        default=False,
                        help="Whether to compile the model state.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/pretrained_models/ESRGAN_x4-DFO2K-25393df7.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--half",
                        action="store_true",
                        help="Use half precision.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="Device to run model.")
    args = parser.parse_args()

    main(args)
