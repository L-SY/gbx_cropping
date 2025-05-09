import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms

class InnerBlackBorderAdder(object):
    def __init__(self, border_width=15):
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

class FixedRotation(object):
    def __init__(self, p=0.75):
        self.p = p
        self.angles = [90, 180, 270]

    def __call__(self, img):
        if torch.rand(1) < self.p:
            angle = self.angles[torch.randint(0, len(self.angles), (1,)).item()]
            return img.rotate(angle)
        return img

class AdaptiveEdgeEnhancer(object):
    def __init__(self, alpha=1.5, beta=0.5, p=0.7):
        self.alpha = alpha
        self.beta = beta
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel, iterations=1)
            combined_edges = cv2.bitwise_or(binary, edges)

            if len(img_np.shape) == 3:
                edge_mask = combined_edges / 255.0
                edge_mask_3d = np.stack([edge_mask] * 3, axis=2)
                sharpened = img_np.astype(float)
                blurred = cv2.GaussianBlur(img_np, (0, 0), 3)
                sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)
                result = img_np * self.beta + sharpened * (1 - self.beta)
                result = result * (1 - edge_mask_3d * self.alpha) + sharpened * (edge_mask_3d * self.alpha)
                result = np.clip(result, 0, 255).astype(np.uint8)
                return Image.fromarray(result)
            else:
                sharpened = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
                result = gray * self.beta + sharpened * (1 - self.beta)
                result = result * (1 - edge_mask * self.alpha) + sharpened * (edge_mask * self.alpha)
                result = np.clip(result, 0, 255).astype(np.uint8)
                return Image.fromarray(result)

        return img

class ContrastTextureEnhancer(object):
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8), p=0.7):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            img_np = np.array(img)
            if len(img_np.shape) == 3:
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                return Image.fromarray(enhanced_rgb)
            else:
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                enhanced = clahe.apply(img_np)
                return Image.fromarray(enhanced)

        return img

def get_training_transform(border_width=70):
    return transforms.Compose([
        FixedRotation(p=0.75),
        InnerBlackBorderAdder(border_width=border_width),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def get_validation_transform(border_width=70):
    return transforms.Compose([
        FixedRotation(p=0.75),
        InnerBlackBorderAdder(border_width=border_width),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def get_inference_transform(border_width=70):
    return transforms.Compose([
        InnerBlackBorderAdder(border_width=border_width),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
