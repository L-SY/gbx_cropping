import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import time
import cv2
import matplotlib.pyplot as plt

class ImageRegressor(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=False):
        super(ImageRegressor, self).__init__()
        if backbone == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

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

        # 保存特征图的钩子
        self.features = {}
        self.register_hooks()

    def register_hooks(self):
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook

        # 注册各层的钩子
        self.backbone.layer1.register_forward_hook(get_features('layer1'))
        self.backbone.layer2.register_forward_hook(get_features('layer2'))
        self.backbone.layer3.register_forward_hook(get_features('layer3'))
        self.backbone.layer4.register_forward_hook(get_features('layer4'))

    def forward(self, x):
        return self.backbone(x).squeeze()

def generate_attention_map(model, image_tensor, save_path):
    """生成注意力图"""
    try:
        # 获取模型输出和特征图
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            # 使用最后一层的特征图
            feature_map = model.features['layer4']

            # 计算特征图的空间平均值作为注意力图
            attention_map = feature_map.mean(dim=1).squeeze()

            # 归一化注意力图
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

            # 转换为numpy并调整大小
            attention_map = attention_map.cpu().numpy()
            attention_map = cv2.resize(attention_map, (224, 224))

            # 读取原始图像
            img = image_tensor.permute(1, 2, 0).cpu().numpy()
            img = ((img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)

            # 创建热力图
            heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)

            # 叠加热力图到原始图像
            superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

            # 保存结果
            cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"生成注意力图时出错: {str(e)}")

def visualize_features(features, save_path):
    """可视化特征图"""
    plt.figure(figsize=(15, 10))
    for idx, (name, feature) in enumerate(features.items(), 1):
        # 获取第一个样本的特征图
        feature = feature[0]  # (C, H, W)

        # 计算特征图的平均值作为一个通道
        feature_mean = feature.mean(dim=0)  # (H, W)

        plt.subplot(2, 2, idx)
        plt.imshow(feature_mean.cpu().numpy(), cmap='viridis')
        plt.title(f'Feature Map: {name}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_model(model_path, device, backbone='resnet34'):
    """加载训练好的模型权重并返回模型"""
    try:
        model = ImageRegressor(backbone=backbone, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise

def predict_single_image(model, device, image_path):
    """对单张图片进行推理并可视化"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到图片: {image_path}")

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
    input_tensor = input_tensor.to(device)

    # 创建可视化输出目录
    vis_dir = 'visualization_output'
    os.makedirs(vis_dir, exist_ok=True)

    # 推理
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        prediction = output.item()

        # 保存特征图可视化
        visualize_features(
            model.features,
            os.path.join(vis_dir, 'feature_maps.png')
        )

        # 生成并保存注意力图
        generate_attention_map(
            model,
            input_tensor,
            os.path.join(vis_dir, 'attention_map.png')
        )

    return prediction

def main():
    parser = argparse.ArgumentParser(description="Predict and visualize with the regression model.")
    parser.add_argument("--image", type=str, required=True,
                        help="输入需要预测的图片路径")
    parser.add_argument("--model", type=str, default="./checkpoints/best_model.pth",
                        help="模型权重文件路径")
    parser.add_argument("--backbone", type=str, default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="骨干网络类型")
    args = parser.parse_args()

    device = torch.device('cpu')
    print("--------------------------------------")

    start_load = time.time()
    model = load_model(args.model, device, backbone=args.backbone)
    end_load = time.time()
    print(f"模型加载耗时: {end_load - start_load:.4f} 秒")

    start_infer = time.time()
    prediction_value = predict_single_image(model, device, args.image)
    end_infer = time.time()
    print(f"推理耗时: {end_infer - start_infer:.4f} 秒")

    print("--------------------------------------")
    print(f"图片: {args.image}")
    print(f"预测值: {prediction_value:.4f}")
    print("可视化结果已保存到 visualization_output 目录")
    print("--------------------------------------")

if __name__ == "__main__":
    main()