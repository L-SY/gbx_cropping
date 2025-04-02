# utils.py
import os
import json
import argparse
import numpy as np
import torch
from PIL import Image

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train a regression model for texture analysis')

    # 主要路径参数
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='输入数据集路径 (包含所有图片和labels.csv的目录)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出路径 (用于保存分割数据集、增强数据集、模型和结果)')

    # 数据分割参数
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例 (默认: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例 (默认: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例 (默认: 0.15)')

    # 数据增强参数
    parser.add_argument('--train_aug_factor', type=int, default=5,
                        help='训练集增强因子 (默认: 15)')
    parser.add_argument('--val_aug_factor', type=int, default=5,
                        help='验证集增强因子 (默认: 15)')
    parser.add_argument('--test_aug_factor', type=int, default=5,
                        help='测试集增强因子 (默认: 5)')
    parser.add_argument('--border_width', type=int, default=70,
                        help='边框宽度 (默认: 70)')

    # 模型相关参数
    parser.add_argument('--backbone', type=str, default='densenet121',
                        choices=['densenet121', 'densenet169', 'resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                        help='主干网络选择 (默认: densenet121)')
    parser.add_argument('--initial_value', type=float, default=15.0,
                        help='回归初始值 (默认: 15.0)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout率 (默认: 0.5)')

    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小 (默认: 32)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='训练轮数 (默认: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--unfreeze_epoch', type=int, default=80,
                        help='解冻主干网络最后几层的轮数 (默认: 80)')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='每隔多少轮评估一次 (默认: 1)')
    parser.add_argument('--patience', type=int, default=80,
                        help='早停耐心值 (默认: 80)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数 (默认: 4)')

    # 其他参数
    parser.add_argument('--no_split', action='store_true',
                        help='跳过数据集分割步骤')
    parser.add_argument('--no_augment', action='store_true',
                        help='跳过数据增强步骤')
    parser.add_argument('--no_train', action='store_true',
                        help='跳过训练步骤')
    parser.add_argument('--no_eval', action='store_true',
                        help='跳过评估步骤')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA，即使可用')

    return parser.parse_args()

def set_seeds(seed=42):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 更多的确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(config, save_path):
    """保存配置文件"""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存到 {save_path}")

def generate_inference_code(backbone, experiment_dir):
    """生成推理示例代码"""
    inference_code = f"""
# 推理示例代码
import torch
from torchvision import transforms
from PIL import Image, ImageDraw

class InnerBlackBorderAdder(object):
    \"\"\"在图像内部添加黑色边框\"\"\"
    def __init__(self, border_width=70):
        self.border_width = border_width

    def __call__(self, img):
        width, height = img.size
        bordered_img = img.copy()
        draw = ImageDraw.Draw(bordered_img)

        draw.rectangle([(0, 0), (width, self.border_width)], fill="black")
        draw.rectangle([(0, height - self.border_width), (width, height)], fill="black")
        draw.rectangle([(0, 0), (self.border_width, height)], fill="black")
        draw.rectangle([(width - self.border_width, 0), (width, height)], fill="black")

        return bordered_img

def predict_image(model_path, image_path, device='cuda'):
    # 加载模型
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型实例
    from models import FrozenCNNRegressor  # 导入你的模型类
    model = FrozenCNNRegressor(backbone='{backbone}', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        InnerBlackBorderAdder(border_width=70),  # 添加与训练时相同的黑色边框
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 执行推理
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return prediction

# 使用示例
# prediction = predict_image('best_model.pth', 'test_image.jpg')
# print(f"预测值: {{prediction:.2f}}")
"""

    with open(os.path.join(experiment_dir, 'inference_example.py'), 'w') as f:
        f.write(inference_code)
    print(f"推理示例代码已保存到 {os.path.join(experiment_dir, 'inference_example.py')}")