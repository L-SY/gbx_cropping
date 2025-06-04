import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

def show_transform_ranges(image_path, save_path=None):
    """
    清晰展示随机变换的最小值、最大值效果
    """
    original_img = Image.open(image_path).convert('RGB')

    # 创建对比图
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Random Transform Ranges - Min vs Max Effects', fontsize=16, fontweight='bold')

    # 原图
    axes[0, 1].imshow(original_img)
    axes[0, 1].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 高斯模糊范围对比
    blur_min = original_img.filter(ImageFilter.GaussianBlur(radius=0.1))
    blur_max = original_img.filter(ImageFilter.GaussianBlur(radius=2.0))

    axes[1, 0].imshow(blur_min)
    axes[1, 0].set_title('Gaussian Blur MIN\nσ = 0.1 (Almost No Blur)',
                         fontsize=11, color='green', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 2].imshow(blur_max)
    axes[1, 2].set_title('Gaussian Blur MAX\nσ = 2.0 (Heavy Blur)',
                         fontsize=11, color='red', fontweight='bold')
    axes[1, 2].axis('off')

    # 亮度范围对比
    brightness_min = ImageEnhance.Brightness(original_img).enhance(0.8)
    brightness_max = ImageEnhance.Brightness(original_img).enhance(1.2)

    axes[2, 0].imshow(brightness_min)
    axes[2, 0].set_title('Brightness MIN\nFactor = 0.8 (Darker)',
                         fontsize=11, color='green', fontweight='bold')
    axes[2, 0].axis('off')

    axes[2, 2].imshow(brightness_max)
    axes[2, 2].set_title('Brightness MAX\nFactor = 1.2 (Brighter)',
                         fontsize=11, color='red', fontweight='bold')
    axes[2, 2].axis('off')

    # 旋转效果（中间列展示）
    rotate_180 = original_img.rotate(180)
    axes[1, 1].imshow(rotate_180)
    axes[1, 1].set_title('Rotation Example\n180° (Most Common)',
                         fontsize=11, color='blue', fontweight='bold')
    axes[1, 1].axis('off')

    # 最极端组合对比
    extreme_light = original_img.copy()
    extreme_light = ImageEnhance.Brightness(extreme_light).enhance(1.2)  # 最亮
    extreme_light = extreme_light.filter(ImageFilter.GaussianBlur(radius=0.1))  # 最清晰

    extreme_dark = original_img.copy()
    extreme_dark = ImageEnhance.Brightness(extreme_dark).enhance(0.8)  # 最暗
    extreme_dark = extreme_dark.filter(ImageFilter.GaussianBlur(radius=2.0))  # 最模糊

    axes[2, 1].imshow(extreme_light)
    axes[2, 1].set_title('BEST Case\n(Bright + Sharp)',
                         fontsize=11, color='green', fontweight='bold')
    axes[2, 1].axis('off')

    axes[0, 0].imshow(extreme_dark)
    axes[0, 0].set_title('WORST Case\n(Dark + Blurry)',
                         fontsize=11, color='red', fontweight='bold')
    axes[0, 0].axis('off')

    # 添加参数范围说明
    info_text = """
PARAMETER RANGES:

🔸 Gaussian Blur:
   σ ∈ [0.1, 2.0]
   Probability: 50%

🔸 Brightness:
   Factor ∈ [0.8, 1.2]
   Probability: 50%

🔸 Rotation:
   Angles: [90°, 180°, 270°]
   Probability: 75%

🔸 Border:
   Fixed: 10% of image size
   Always applied
    """

    axes[0, 2].text(0.05, 0.95, info_text, transform=axes[0, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[0, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Range comparison saved to: {save_path}")
    else:
        plt.show()

def print_range_summary():
    """
    打印变换范围的详细说明
    """
    print("=" * 60)
    print("RANDOM TRANSFORM RANGES SUMMARY")
    print("=" * 60)

    transforms_info = [
        {
            "name": "Gaussian Blur",
            "parameter": "sigma (radius)",
            "min_val": 0.1,
            "max_val": 2.0,
            "probability": "50%",
            "effect_min": "Almost no blur, image stays sharp",
            "effect_max": "Heavy blur, details become unclear"
        },
        {
            "name": "Brightness",
            "parameter": "enhancement factor",
            "min_val": 0.8,
            "max_val": 1.2,
            "probability": "50%",
            "effect_min": "Image becomes 20% darker",
            "effect_max": "Image becomes 20% brighter"
        },
        {
            "name": "Rotation",
            "parameter": "angle (degrees)",
            "min_val": "90°",
            "max_val": "270°",
            "probability": "75%",
            "effect_min": "90° clockwise rotation",
            "effect_max": "270° clockwise rotation"
        },
        {
            "name": "Black Border",
            "parameter": "border percentage",
            "min_val": "10%",
            "max_val": "10%",
            "probability": "100%",
            "effect_min": "Fixed border size",
            "effect_max": "Fixed border size"
        }
    ]

    for t in transforms_info:
        print(f"\n📊 {t['name']}:")
        print(f"   Parameter: {t['parameter']}")
        print(f"   Range: {t['min_val']} → {t['max_val']}")
        print(f"   Probability: {t['probability']}")
        print(f"   Min Effect: {t['effect_min']}")
        print(f"   Max Effect: {t['effect_max']}")

    print("\n" + "=" * 60)
    print("IMPACT ON YOUR FOAM DENSITY PREDICTION:")
    print("=" * 60)
    print("🔴 HIGH IMPACT (May cause prediction errors):")
    print("   • Brightness: 0.8-1.2 range may significantly change foam appearance")
    print("   • Heavy blur (σ=2.0): May lose important texture details")
    print("\n🟡 MEDIUM IMPACT:")
    print("   • Rotation: Changes perspective but preserves foam structure")
    print("   • Light blur (σ=0.1): Minimal impact on foam texture")
    print("\n🟢 LOW IMPACT:")
    print("   • Black border: Should not affect foam density analysis")
    print("\n💡 RECOMMENDATIONS:")
    print("   • Consider narrower brightness range: [0.9, 1.1]")
    print("   • Consider lower max blur: σ_max = 1.0")
    print("   • Keep rotation as-is (good for data augmentation)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Show random transform ranges")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save comparison")
    parser.add_argument("--summary", action="store_true", help="Print detailed range summary")

    args = parser.parse_args()

    # 显示可视化对比
    show_transform_ranges(args.image_path, args.save_path)

    # 显示详细统计
    if args.summary:
        print_range_summary()