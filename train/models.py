# models.py
import torch
import torch.nn as nn
from torchvision import models

class FrozenCNNRegressor(nn.Module):
    """使用冻结CNN特征提取器和可训练FC层的纹理回归模型"""
    def __init__(self, backbone='densenet121', pretrained=True, initial_value=15.0, dropout_rate=0.5):
        super(FrozenCNNRegressor, self).__init__()

        # 加载预训练的骨干网络
        if backbone == 'densenet121':
            base_model = models.densenet121(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            feature_dim = base_model.classifier.in_features  # 1024
        elif backbone == 'densenet169':
            base_model = models.densenet169(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            feature_dim = base_model.classifier.in_features  # 1664
        elif backbone == 'resnet18':
            base_model = models.resnet18(weights='DEFAULT' if pretrained else None)
            # 移除全局平均池化和全连接层
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(weights='DEFAULT' if pretrained else None)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 2048
        elif backbone == 'mobilenet_v2':
            base_model = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            feature_dim = 1280
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")

        # 冻结特征提取器
        for param in self.features.parameters():
            param.requires_grad = False

        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 带有L2正则化效果的回归头 (类似于Ridge回归)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(64, 1)
        )

        # 将最终层的偏置初始化为指定值
        final_layer = self.regressor[-1]
        nn.init.constant_(final_layer.bias, initial_value)

    def forward(self, x):
        # 提取特征 (冻结阶段)
        with torch.no_grad():
            features = self.features(x)

        # 全局平均池化
        pooled = self.global_pool(features)

        # 回归预测 (可训练部分)
        output = self.regressor(pooled).squeeze()

        return output

    def unfreeze_last_layers(self, num_layers=2):
        """解冻CNN特征提取器的最后几层进行微调"""
        # 实现取决于特定的骨干网络
        if isinstance(self.features, nn.Sequential):
            # 这适用于ResNet和其他顺序模型
            for i, module in enumerate(list(self.features.children())[-num_layers:]):
                for param in module.parameters():
                    param.requires_grad = True
            print(f"解冻最后 {num_layers} 个顺序模块")
        elif hasattr(self.features, 'denseblock4'):
            # 这适用于DenseNet
            for param in self.features.denseblock4.parameters():
                param.requires_grad = True
            for param in self.features.norm5.parameters():
                param.requires_grad = True
            print(f"解冻DenseNet的最后一个密集块和标准化层")
        else:
            print("未知的骨干网络结构，未解冻任何层")