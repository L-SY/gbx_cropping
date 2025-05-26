import random
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from torchvision import transforms

class InnerBlackBorderAdder(object):
    def __init__(self, border_percentage=0.1):
        self.border_percentage = border_percentage

    def __call__(self, img):
        width, height = img.size
        bordered_img = img.copy()
        draw = ImageDraw.Draw(bordered_img)

        border_width_h = int(width * self.border_percentage)
        border_width_v = int(height * self.border_percentage)

        draw.rectangle([(0, 0), (width, border_width_v)], fill="black")
        draw.rectangle([(0, height - border_width_v), (width, height)], fill="black")
        draw.rectangle([(0, 0), (border_width_h, height)], fill="black")
        draw.rectangle([(width - border_width_h, 0), (width, height)], fill="black")

        return bordered_img


class RandomGaussianBlur(object):
    def __init__(self, p=0.5, sigma_min=0.1, sigma_max=2.0):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

class RandomBrightness(object):
    def __init__(self, p=0.5, brightness_range=(0.8, 1.2)):
        self.p = p
        self.brightness_range = brightness_range

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(factor)
        return img

class FixedRotation(object):
    def __init__(self, p=0.75):
        self.p = p
        self.angles = [90, 180, 270]

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            angle = random.choice(self.angles)
            return img.rotate(angle)
        return img

# (Other enhancers like AdaptiveEdgeEnhancer, ContrastTextureEnhancer can remain unchanged)

# Example transform compositions

def get_training_transform(border_percentage=0.1):
    return transforms.Compose([
        FixedRotation(p=0.75),
        InnerBlackBorderAdder(border_percentage=border_percentage),
        RandomGaussianBlur(p=0.5, sigma_min=0.1, sigma_max=2.0),
        RandomBrightness(p=0.5, brightness_range=(0.8, 1.2)),
        transforms.ToTensor(),
    ])


def get_validation_transform(border_percentage=0.1):
    return transforms.Compose([
        FixedRotation(p=0.75),
        InnerBlackBorderAdder(border_percentage=border_percentage),
        transforms.ToTensor(),
    ])


def get_inference_transform(border_percentage=0.1):
    return transforms.Compose([
        InnerBlackBorderAdder(border_percentage=border_percentage),
        transforms.ToTensor(),
    ])
