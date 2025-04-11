# transforms.py
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms

class InnerBlackBorderAdder(object):
    """在图像内部添加黑色边框"""
    def __init__(self, border_width=15):
        self.border_width = border_width

    def __call__(self, img):
        # 直接应用边框，不进行随机性检查
        width, height = img.size
        bordered_img = img.copy()
        draw = ImageDraw.Draw(bordered_img)

        draw.rectangle([(0, 0), (width, self.border_width)], fill="black")
        draw.rectangle([(0, height - self.border_width), (width, height)], fill="black")
        draw.rectangle([(0, 0), (self.border_width, height)], fill="black")
        draw.rectangle([(width - self.border_width, 0), (width, height)], fill="black")

        return bordered_img

class FixedRotation(object):
    """固定角度旋转: 仅旋转 0°, 90°, 180° 或 270°"""
    def __init__(self, p=0.75):
        """
        参数:
            p: 应用旋转的概率 (0° 角度的概率为 1-p)
        """
        self.p = p
        self.angles = [90, 180, 270]  # 可能的旋转角度

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # 随机选择一个角度
            angle = self.angles[torch.randint(0, len(self.angles), (1,)).item()]
            return img.rotate(angle)
        return img  # 不旋转 (0°)

class AdaptiveEdgeEnhancer(object):
    """自适应边缘增强器"""
    def __init__(self, alpha=1.5, beta=0.5, p=0.7):
        """
        参数:
            alpha: 边缘增强强度
            beta: 原始图像保留比例
            p: 应用变换的概率
        """
        self.alpha = alpha
        self.beta = beta
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # 转换为numpy数组
            img_np = np.array(img)

            # 转换为灰度图用于边缘检测
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np

            # 使用自适应阈值方法 - 高斯权重，块大小11，常数2
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # 使用Canny算法进一步进行边缘检测
            edges = cv2.Canny(gray, 50, 150)

            # 应用形态学操作连接附近的边缘
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel, iterations=1)

            # 合并两种边缘效果
            combined_edges = cv2.bitwise_or(binary, edges)

            # 对于彩色图像，使用边缘增强
            if len(img_np.shape) == 3:
                # 创建边缘掩码
                edge_mask = combined_edges / 255.0
                edge_mask_3d = np.stack([edge_mask] * 3, axis=2)

                # 锐化原始图像
                sharpened = img_np.astype(float)
                blurred = cv2.GaussianBlur(img_np, (0, 0), 3)
                sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)

                # 混合原始图像和边缘信息
                result = img_np * self.beta + sharpened * (1 - self.beta)
                # 在边缘位置额外增强
                result = result * (1 - edge_mask_3d * self.alpha) + sharpened * (edge_mask_3d * self.alpha)
                result = np.clip(result, 0, 255).astype(np.uint8)

                return Image.fromarray(result)

            else:
                # 灰度处理
                sharpened = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
                result = gray * self.beta + sharpened * (1 - self.beta)
                result = result * (1 - edge_mask * self.alpha) + sharpened * (edge_mask * self.alpha)
                result = np.clip(result, 0, 255).astype(np.uint8)

                return Image.fromarray(result)

        return img

class ContrastTextureEnhancer(object):
    """对比度感知的纹理增强器"""
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8), p=0.7):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # 转换为numpy数组
            img_np = np.array(img)

            # 转换为LAB色彩空间
            if len(img_np.shape) == 3:  # 彩色图像
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                # 对L通道应用CLAHE (对比度受限的自适应直方图均衡化)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                cl = clahe.apply(l)

                # 合并回LAB并转换为RGB
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

                return Image.fromarray(enhanced_rgb)
            else:  # 灰度图
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                enhanced = clahe.apply(img_np)
                return Image.fromarray(enhanced)

        return img

def get_training_transform(border_width=70):
    """获取训练数据转换"""
    return transforms.Compose([
        # 1) 固定角度旋转
        FixedRotation(p=0.75),
        # # 2) 自适应边缘增强
        # AdaptiveEdgeEnhancer(alpha=1.7, beta=0.4, p=0.8),
        # # 3) 对比度感知的纹理增强
        # ContrastTextureEnhancer(clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),
        InnerBlackBorderAdder(border_width=border_width),
        # 调整为目标大小
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def get_validation_transform(border_width=70):
    """获取验证/测试数据转换"""
    return transforms.Compose([
        # 仅使用基本预处理进行验证/测试
        FixedRotation(p=0.75),
        InnerBlackBorderAdder(border_width=border_width),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def get_inference_transform(border_width=70):
    """获取推理时的转换"""
    return transforms.Compose([
        InnerBlackBorderAdder(border_width=border_width),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])