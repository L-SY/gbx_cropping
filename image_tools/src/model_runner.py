import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import time

# ===================== 1. 定义与训练阶段一致的网络结构 ===================== #
class ImageRegressor(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=False):
        super(ImageRegressor, self).__init__()
        # 你在训练时使用的 backbone，若要加载官方预训练权重可设 pretrained=True
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError("Unsupported backbone model")

        # 修改最后一层，用于单输出回归
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze()


def load_model(model_path, device, backbone='resnet34'):
    """
    加载训练好的模型权重并返回模型
    """
    model = ImageRegressor(backbone=backbone, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # 注意检查 checkpoint 的结构
    # 如果是 state_dict 需要用load_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # 如果 checkpoint 就是模型权重本身
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def predict_single_image(model, device, image_path):
    """
    对单张图片进行推理并返回预测值
    """
    # 确保图像路径存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到图片: {image_path}")

    # 与训练数据一致的 transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 读取图片和预处理
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    # 增加batch维度 => (1, C, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        # output 可能是一个标量tensor
        prediction = output.item()

    return prediction


def main():
    """
    主函数：解析命令行参数并执行预测
    """
    parser = argparse.ArgumentParser(description="Predict single image with the best regression model.")
    parser.add_argument("--image", type=str, required=True,
                        help="输入需要预测的图片路径，比如 /path/to/image.jpg")
    parser.add_argument("--model", type=str, default="./checkpoints/best_model.pth",
                        help="已训练好的模型权重文件路径，默认 ./checkpoints/best_model.pth")
    parser.add_argument("--backbone", type=str, default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="指定加载模型时使用的骨干网络(需与训练时一致)，默认resnet34")
    args = parser.parse_args()

    # 选择设备
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("--------------------------------------")
    start_load = time.time()
    # 加载模型
    model = load_model(args.model, device, backbone=args.backbone)
    end_load = time.time()
    print(f"模型加载耗时: {end_load - start_load:.4f} 秒")

    # 进行预测
    start_infer = time.time()
    prediction_value = predict_single_image(model, device, args.image)
    end_infer = time.time()
    print(f"推理耗时: {end_infer - start_infer:.4f} 秒")

    # 输出预测
    print("--------------------------------------")
    print(f"图片: {args.image}")
    print(f"预测值: {prediction_value:.4f}")
    print("--------------------------------------")


if __name__ == "__main__":
    main()