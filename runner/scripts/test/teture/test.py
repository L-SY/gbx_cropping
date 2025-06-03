import cv2
import numpy as np
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage
import os

class TextureExtractor:
    def __init__(self, image_path):
        """
        初始化纹路提取器
        Args:
            image_path: 输入图片路径
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def extract_edges(self, method='canny'):
        """
        提取边缘信息（纹路的基础）
        Args:
            method: 边缘检测方法 ('canny', 'sobel', 'laplacian')
        """
        if method == 'canny':
            edges = cv2.Canny(self.gray, 50, 150)
        elif method == 'sobel':
            sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        elif method == 'laplacian':
            edges = cv2.Laplacian(self.gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))

        return edges

    def extract_lbp_texture(self, radius=3, n_points=24):
        """
        使用局部二值模式(LBP)提取纹理特征
        Args:
            radius: LBP半径
            n_points: 采样点数
        """
        def lbp(image, radius, n_points):
            h, w = image.shape
            lbp_image = np.zeros((h, w), dtype=np.uint8)

            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = image[i, j]
                    binary_string = ''

                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))

                        if 0 <= x < h and 0 <= y < w:
                            if image[x, y] >= center:
                                binary_string += '1'
                            else:
                                binary_string += '0'

                    lbp_image[i, j] = int(binary_string, 2)

            return lbp_image

        return lbp(self.gray, radius, n_points)

    def extract_gabor_texture(self):
        """
        使用Gabor滤波器提取纹理特征
        """
        gabor_responses = []

        # 不同方向和频率的Gabor滤波器
        for theta in [0, 45, 90, 135]:  # 角度
            for frequency in [0.1, 0.3, 0.5]:  # 频率
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta),
                                            2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(self.gray, cv2.CV_8UC3, kernel)
                gabor_responses.append(filtered)

        # 合并所有Gabor响应
        gabor_texture = np.mean(gabor_responses, axis=0)
        return np.uint8(gabor_texture)

    def extract_main_texture_patterns(self, patch_size=16, n_clusters=8):
        """
        提取主要纹理模式
        Args:
            patch_size: 纹理块大小
            n_clusters: 聚类数量
        """
        # 将图像分割成小块
        patches = image.extract_patches_2d(self.gray, (patch_size, patch_size))

        # 将每个patch展平为特征向量
        patch_features = patches.reshape(patches.shape[0], -1)

        # 使用K-means聚类找到主要纹理模式
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(patch_features)

        # 重构聚类中心作为主要纹理模式
        texture_patterns = kmeans.cluster_centers_.reshape(n_clusters, patch_size, patch_size)

        return texture_patterns, cluster_labels

    def extract_dominant_directions(self):
        """
        提取主导纹理方向
        """
        # 计算梯度
        grad_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值和方向
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # 将方向转换为0-180度
        direction_degrees = np.degrees(direction) % 180

        # 统计方向直方图
        hist, bins = np.histogram(direction_degrees, bins=18, range=(0, 180),
                                  weights=magnitude)

        # 找到主导方向
        dominant_angles = bins[:-1][hist > np.mean(hist)]

        return direction_degrees, dominant_angles, hist

    def visualize_results(self, save_path=None):
        """
        可视化所有纹理提取结果
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Edge detection
        edges_canny = self.extract_edges('canny')
        axes[0, 1].imshow(edges_canny, cmap='gray')
        axes[0, 1].set_title('Canny Edge Detection')
        axes[0, 1].axis('off')

        edges_sobel = self.extract_edges('sobel')
        axes[0, 2].imshow(edges_sobel, cmap='gray')
        axes[0, 2].set_title('Sobel Edge Detection')
        axes[0, 2].axis('off')

        # LBP texture
        lbp_texture = self.extract_lbp_texture()
        axes[1, 0].imshow(lbp_texture, cmap='gray')
        axes[1, 0].set_title('LBP Texture Features')
        axes[1, 0].axis('off')

        # Gabor texture
        gabor_texture = self.extract_gabor_texture()
        axes[1, 1].imshow(gabor_texture, cmap='gray')
        axes[1, 1].set_title('Gabor Texture Features')
        axes[1, 1].axis('off')

        # Main texture patterns
        texture_patterns, _ = self.extract_main_texture_patterns()
        # Display first 4 main patterns
        combined_patterns = np.hstack([texture_patterns[i] for i in range(min(4, len(texture_patterns)))])
        axes[1, 2].imshow(combined_patterns, cmap='gray')
        axes[1, 2].set_title('Main Texture Patterns')
        axes[1, 2].axis('off')

        # Direction distribution
        direction_map, dominant_angles, hist = self.extract_dominant_directions()
        axes[2, 0].imshow(direction_map, cmap='hsv')
        axes[2, 0].set_title('Texture Direction Distribution')
        axes[2, 0].axis('off')

        # Direction histogram
        angles = np.arange(0, 180, 10)
        axes[2, 1].bar(angles, hist)
        axes[2, 1].set_title('Dominant Texture Directions')
        axes[2, 1].set_xlabel('Angle (degrees)')
        axes[2, 1].set_ylabel('Intensity')

        # Combined texture information
        # Combine edge and texture features
        combined = 0.4 * edges_canny + 0.3 * lbp_texture + 0.3 * gabor_texture
        axes[2, 2].imshow(combined, cmap='gray')
        axes[2, 2].set_title('Combined Texture Information')
        axes[2, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'edges_canny': edges_canny,
            'edges_sobel': edges_sobel,
            'lbp_texture': lbp_texture,
            'gabor_texture': gabor_texture,
            'texture_patterns': texture_patterns,
            'direction_map': direction_map,
            'dominant_angles': dominant_angles,
            'combined_texture': combined
        }

def main():
    """
    Main function - Usage example
    """
    # Input image path
    image_path = "/runner/scripts/test/teture/train.jpg"

    if not os.path.exists(image_path):
        print("Image file does not exist!")
        return

    try:
        # Create texture extractor
        extractor = TextureExtractor(image_path)

        print("Extracting texture information...")

        # Extract and visualize results
        results = extractor.visualize_results()

        # Save results
        save_option = input("Save result image? (y/n): ")
        if save_option.lower() == 'y':
            save_path = f"texture_analysis_{os.path.basename(image_path)}"
            extractor.visualize_results(save_path)
            print(f"Results saved as: {save_path}")

        # Output main texture information statistics
        print("\n=== Texture Analysis Results ===")
        print(f"Dominant texture directions: {results['dominant_angles']} degrees")
        print(f"Detected {len(results['texture_patterns'])} main texture patterns")

        # Texture complexity assessment
        edge_density = np.sum(results['edges_canny'] > 0) / results['edges_canny'].size
        print(f"Edge density: {edge_density:.3f}")

        texture_variance = np.var(results['lbp_texture'])
        print(f"Texture variation: {texture_variance:.2f}")

    except Exception as e:
        print(f"Error occurred during processing: {e}")

if __name__ == "__main__":
    main()