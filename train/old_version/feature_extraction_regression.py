# 导入必要的库
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TextureDataset(Dataset):
    """纹理数据集"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def extract_glcm_features(img):
    """提取GLCM纹理特征"""
    # 确保图像是灰度图
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 标准化灰度值到0-255
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # 计算GLCM
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    # 提取GLCM属性
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []

    for prop in properties:
        if prop == 'ASM':  # Angular Second Moment
            # ASM is the sum of squared elements in the GLCM
            for d in range(len(distances)):
                for a in range(len(angles)):
                    features.append(np.sum(glcm[:, :, d, a] ** 2))
        else:
            glcm_feature = graycoprops(glcm, prop).ravel()
            features.extend(glcm_feature)

    return np.array(features)

def extract_lbp_features(img, P=8, R=1):
    """提取局部二值模式特征"""
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算LBP
    lbp = local_binary_pattern(img, P=P, R=R, method='uniform')

    # 计算直方图
    n_bins = P + 2  # uniform pattern
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1),
                           range=(0, n_bins), density=True)

    return hist

def extract_statistical_features(img):
    """提取统计特征"""
    if len(img.shape) > 2:
        channels = cv2.split(img)
        features = []

        for channel in channels:
            # 基本统计量
            mean = np.mean(channel)
            std = np.std(channel)
            median = np.median(channel)
            min_val = np.min(channel)
            max_val = np.max(channel)

            # 熵
            entropy = shannon_entropy(channel)

            # 分位数
            q1 = np.percentile(channel, 25)
            q3 = np.percentile(channel, 75)

            channel_features = [mean, std, median, min_val, max_val, entropy, q1, q3]
            features.extend(channel_features)
    else:
        # 单通道图像
        mean = np.mean(img)
        std = np.std(img)
        median = np.median(img)
        min_val = np.min(img)
        max_val = np.max(img)
        entropy = shannon_entropy(img)
        q1 = np.percentile(img, 25)
        q3 = np.percentile(img, 75)

        features = [mean, std, median, min_val, max_val, entropy, q1, q3]

    return np.array(features)

def extract_combined_texture_features(image_path):
    """结合多种方法提取纹理特征"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 1. 提取GLCM特征
        glcm_features = extract_glcm_features(img)

        # 2. 提取LBP特征 - 使用不同半径和邻域点数
        lbp_features1 = extract_lbp_features(img, P=8, R=1)
        lbp_features2 = extract_lbp_features(img, P=16, R=2)
        lbp_features3 = extract_lbp_features(img, P=24, R=3)

        # 3. 提取统计特征
        stat_features = extract_statistical_features(img)

        # 4. 提取频域特征 (FFT特征)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        f_transform = np.fft.fft2(gray_img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

        # 计算频谱的统计特征
        fft_mean = np.mean(magnitude_spectrum)
        fft_std = np.std(magnitude_spectrum)
        fft_features = [fft_mean, fft_std]

        # 结合所有特征
        all_features = np.concatenate([
            glcm_features,
            lbp_features1,
            lbp_features2,
            lbp_features3,
            stat_features,
            fft_features
        ])

        return all_features

    except Exception as e:
        print(f"提取特征时出错: {e}")
        # 返回全零向量作为替代
        return np.zeros(100)  # 假设特征维度为100

class SimpleFeatureExtractor:
    """使用预训练CNN提取特征的类"""
    def __init__(self, model_name='resnet18', device='cuda'):
        self.device = device

        # 加载预训练模型
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            self.feature_dim = model.fc.in_features
            self.model = nn.Sequential(*list(model.children())[:-1])  # 移除最后的FC层
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            self.feature_dim = model.classifier.in_features
            self.model = model.features
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            self.feature_dim = model.classifier[1].in_features
            self.model = model.features
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        self.model = self.model.to(device)
        self.model.eval()

        # 预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path):
        """提取单个图像的特征"""
        try:
            # 加载并预处理图像
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # 提取特征
            with torch.no_grad():
                features = self.model(img_tensor)
                # 处理特征以使其成为1D向量
                if len(features.shape) == 4:  # 如果特征是4D的 [B, C, H, W]
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)

            return features.cpu().numpy()[0]

        except Exception as e:
            print(f"提取CNN特征时出错: {e}")
            return np.zeros(self.feature_dim)

def evaluate_regression_models(X, y, model_dict, cv=LeaveOneOut()):
    """评估多个回归模型的性能"""
    results = {}

    for name, model in model_dict.items():
        # 使用均方误差评估
        mse_scores = cross_val_score(model, X, y, cv=cv,
                                     scoring='neg_mean_squared_error')
        rmse = np.sqrt(-mse_scores.mean())
        rmse_std = np.sqrt(mse_scores.std())

        # 使用R²评估
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        r2_mean = r2_scores.mean()
        r2_std = r2_scores.std()

        # 使用MAE评估
        mae_scores = cross_val_score(model, X, y, cv=cv,
                                     scoring='neg_mean_absolute_error')
        mae = -mae_scores.mean()
        mae_std = mae_scores.std()

        results[name] = {
            'rmse': rmse,
            'rmse_std': rmse_std,
            'r2': r2_mean,
            'r2_std': r2_std,
            'mae': mae,
            'mae_std': mae_std
        }

    return results

def get_predictions(X, y, model, cv=LeaveOneOut()):
    """使用交叉验证获取预测值"""
    y_pred = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)

    return y_pred

def plot_predictions(y_true, y_pred, title, save_path=None):
    """绘制真实值与预测值的对比图"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', s=80)

    # 添加完美预测线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # 添加回归线
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), 'g-', lw=1.5, alpha=0.7)

    plt.title(f'{title}\nRMSE = {rmse:.4f}, R² = {r2:.4f}')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.grid(True, alpha=0.3)

    # 设置轴范围
    margin = (max_val - min_val) * 0.05  # 5% margin
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

def feature_importance_plot(X, feature_names, best_model, save_path=None):
    """绘制特征重要性图(仅适用于某些模型)"""
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        indices = np.argsort(importance)[-20:]  # 只展示前20个重要特征

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

def create_ensemble_model(models, X, y, cv=LeaveOneOut()):
    """创建模型集成，使用每个模型在LOOCV中的表现作为权重"""
    model_weights = {}
    model_predictions = {}

    # 为每个模型获取LOOCV预测值和性能分数
    for name, model in models.items():
        y_pred = get_predictions(X, y, model, cv)
        model_predictions[name] = y_pred
        model_weights[name] = r2_score(y, y_pred)

    # 如果有负R²，则将所有权重加上最小值的绝对值+1，确保权重为正
    min_weight = min(model_weights.values())
    if min_weight < 0:
        for name in model_weights:
            model_weights[name] += abs(min_weight) + 1

    # 归一化权重
    total_weight = sum(model_weights.values())
    for name in model_weights:
        model_weights[name] /= total_weight

    # 生成加权集成预测
    ensemble_pred = np.zeros_like(y, dtype=float)
    for name, pred in model_predictions.items():
        ensemble_pred += pred * model_weights[name]

    return ensemble_pred, model_weights

def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置路径
    base_dir = "/home/lsy/gbx_cropping_ws/src/image_tools/only_fifteen/augmented_dataset/train"
    labels_file = os.path.join(base_dir, "train_labels.csv")

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"texture_analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # 加载数据
    df = pd.read_csv(labels_file)
    print(f"Found {len(df)} entries in labels file")

    # 验证图片存在
    valid_entries = []
    for idx, row in df.iterrows():
        img_path = os.path.join(base_dir, row['image_name'])
        if os.path.exists(img_path):
            valid_entries.append({
                'image_path': img_path,
                'label': row['label']
            })
        else:
            print(f"Warning: Image not found: {img_path}")

    print(f"Found {len(valid_entries)} valid images")
    if len(valid_entries) == 0:
        raise ValueError("No valid images found!")

    # 创建图像路径和标签列表
    image_paths = [entry['image_path'] for entry in valid_entries]
    labels = np.array([entry['label'] for entry in valid_entries])

    # 打印标签统计信息
    print(f"Labels statistics: mean={labels.mean():.2f}, std={labels.std():.2f}, "
          f"min={labels.min():.2f}, max={labels.max():.2f}")

    # 保存标签分布图
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=10, alpha=0.7)
    plt.axvline(labels.mean(), color='red', linestyle='dashed', linewidth=2)
    plt.title('Label Distribution')
    plt.xlabel('Label Value')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'label_distribution.png'))
    plt.close()

    print("\n==== 提取传统纹理特征 ====")
    # 提取传统纹理特征
    X_traditional = []
    for img_path in tqdm(image_paths, desc="Extracting traditional features"):
        features = extract_combined_texture_features(img_path)
        X_traditional.append(features)

    X_traditional = np.array(X_traditional)
    print(f"Traditional feature shape: {X_traditional.shape}")

    # 标准化特征
    scaler_trad = StandardScaler()
    X_traditional_scaled = scaler_trad.fit_transform(X_traditional)

    # 生成特征名称(用于可视化)
    feature_names_trad = []
    # GLCM特征
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    distances = [1, 3, 5]
    angles = ['0deg', '45deg', '90deg', '135deg']
    for prop in properties:
        for d in distances:
            for a in angles:
                if prop == 'ASM':
                    feature_names_trad.append(f"GLCM_{prop}_d{d}_{a}")
                else:
                    feature_names_trad.append(f"GLCM_{prop}_d{d}_{a}")

    # LBP特征
    for r, p in [(1, 8), (2, 16), (3, 24)]:
        for i in range(p+2):
            feature_names_trad.append(f"LBP_R{r}P{p}_bin{i}")

    # 统计特征
    for ch in ['R', 'G', 'B']:
        for stat in ['mean', 'std', 'median', 'min', 'max', 'entropy', 'q1', 'q3']:
            feature_names_trad.append(f"{ch}_{stat}")

    # FFT特征
    feature_names_trad.extend(['FFT_mean', 'FFT_std'])

    # 如果特征名称数量与特征不匹配，使用默认名称
    if len(feature_names_trad) != X_traditional.shape[1]:
        print(f"Warning: Feature names count ({len(feature_names_trad)}) doesn't match "
              f"feature count ({X_traditional.shape[1]}). Using default feature names.")
        feature_names_trad = [f"Feature_{i}" for i in range(X_traditional.shape[1])]

    print("\n==== 提取CNN特征 ====")
    # 提取CNN特征
    extractor = SimpleFeatureExtractor(model_name='densenet121', device=device)
    X_cnn = []
    for img_path in tqdm(image_paths, desc="Extracting CNN features"):
        features = extractor.extract_features(img_path)
        X_cnn.append(features)

    X_cnn = np.array(X_cnn)
    print(f"CNN feature shape: {X_cnn.shape}")

    # 标准化特征
    scaler_cnn = StandardScaler()
    X_cnn_scaled = scaler_cnn.fit_transform(X_cnn)

    # 创建特征名称
    feature_names_cnn = [f"CNN_{i}" for i in range(X_cnn.shape[1])]

    print("\n==== 评估传统特征的回归模型 ====")
    # 定义回归模型
    traditional_models = {
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    # 评估传统特征的模型性能
    traditional_results = evaluate_regression_models(X_traditional_scaled, labels,
                                                     traditional_models)

    # 打印结果
    print("Traditional Feature Results:")
    for name, metrics in traditional_results.items():
        print(f"{name}: RMSE = {metrics['rmse']:.4f}±{metrics['rmse_std']:.4f}, "
              f"R² = {metrics['r2']:.4f}±{metrics['r2_std']:.4f}, "
              f"MAE = {metrics['mae']:.4f}±{metrics['mae_std']:.4f}")

    # 找出最佳模型
    best_traditional_model = max(traditional_results.items(),
                                 key=lambda x: x[1]['r2'])[0]
    print(f"Best Traditional Model: {best_traditional_model}")

    # 获取最佳模型的预测结果
    y_pred_trad = get_predictions(X_traditional_scaled, labels,
                                  traditional_models[best_traditional_model])

    # 绘制预测图
    plot_predictions(labels, y_pred_trad, f'Traditional Features - {best_traditional_model}',
                     os.path.join(results_dir, 'traditional_predictions.png'))

    # 绘制特征重要性(如果是随机森林)
    if best_traditional_model == 'RandomForest':
        # 先在全部数据上拟合模型
        traditional_models[best_traditional_model].fit(X_traditional_scaled, labels)
        feature_importance_plot(X_traditional_scaled, feature_names_trad,
                                traditional_models[best_traditional_model],
                                os.path.join(results_dir, 'traditional_feature_importance.png'))

    print("\n==== 评估CNN特征的回归模型 ====")
    # 评估CNN特征的模型性能
    cnn_results = evaluate_regression_models(X_cnn_scaled, labels, traditional_models)

    # 打印结果
    print("CNN Feature Results:")
    for name, metrics in cnn_results.items():
        print(f"{name}: RMSE = {metrics['rmse']:.4f}±{metrics['rmse_std']:.4f}, "
              f"R² = {metrics['r2']:.4f}±{metrics['r2_std']:.4f}, "
              f"MAE = {metrics['mae']:.4f}±{metrics['mae_std']:.4f}")

    # 找出最佳模型
    best_cnn_model = max(cnn_results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"Best CNN Model: {best_cnn_model}")

    # 获取最佳模型的预测结果
    y_pred_cnn = get_predictions(X_cnn_scaled, labels, traditional_models[best_cnn_model])

    # 绘制预测图
    plot_predictions(labels, y_pred_cnn, f'CNN Features - {best_cnn_model}',
                     os.path.join(results_dir, 'cnn_predictions.png'))

    print("\n==== 创建特征融合模型 ====")
    # 融合传统特征和CNN特征
    X_combined = np.hstack((X_traditional_scaled, X_cnn_scaled))
    feature_names_combined = feature_names_trad + feature_names_cnn
    print(f"Combined feature shape: {X_combined.shape}")

    # 评估融合特征的模型性能
    combined_results = evaluate_regression_models(X_combined, labels, traditional_models)

    # 打印结果
    print("Combined Feature Results:")
    for name, metrics in combined_results.items():
        print(f"{name}: RMSE = {metrics['rmse']:.4f}±{metrics['rmse_std']:.4f}, "
              f"R² = {metrics['r2']:.4f}±{metrics['r2_std']:.4f}, "
              f"MAE = {metrics['mae']:.4f}±{metrics['mae_std']:.4f}")

    # 找出最佳模型
    best_combined_model = max(combined_results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"Best Combined Model: {best_combined_model}")

    # 获取最佳模型的预测结果
    y_pred_combined = get_predictions(X_combined, labels,
                                      traditional_models[best_combined_model])

    # 绘制预测图
    plot_predictions(labels, y_pred_combined, f'Combined Features - {best_combined_model}',
                     os.path.join(results_dir, 'combined_predictions.png'))

    # 绘制特征重要性(如果是随机森林)
    if best_combined_model == 'RandomForest':
        # 先在全部数据上拟合模型
        traditional_models[best_combined_model].fit(X_combined, labels)
        feature_importance_plot(X_combined, feature_names_combined,
                                traditional_models[best_combined_model],
                                os.path.join(results_dir, 'combined_feature_importance.png'))

    print("\n==== 创建模型集成 ====")
    # 选择三种不同特征集的最佳模型进行集成
    ensemble_models = {
        f'Trad_{best_traditional_model}': traditional_models[best_traditional_model],
        f'CNN_{best_cnn_model}': traditional_models[best_cnn_model],
        f'Comb_{best_combined_model}': traditional_models[best_combined_model]
    }

    # 对三种不同的特征集分别获取预测结果
    predictions_dict = {
        f'Trad_{best_traditional_model}': y_pred_trad,
        f'CNN_{best_cnn_model}': y_pred_cnn,
        f'Comb_{best_combined_model}': y_pred_combined
    }

    # 计算加权平均预测结果
    # 根据R²值分配权重
    weights = {}
    for name, pred in predictions_dict.items():
        r2 = r2_score(labels, pred)
        weights[name] = max(0, r2)  # 确保权重非负

    # 如果所有权重都为0，则均匀分配
    if sum(weights.values()) == 0:
        for name in weights:
            weights[name] = 1 / len(weights)
    else:
        # 归一化权重
        total_weight = sum(weights.values())
        for name in weights:
            weights[name] /= total_weight

    # 计算加权平均预测
    y_pred_ensemble = np.zeros_like(labels, dtype=float)
    for name, pred in predictions_dict.items():
        y_pred_ensemble += pred * weights[name]

    # 计算集成模型性能
    ensemble_rmse = np.sqrt(mean_squared_error(labels, y_pred_ensemble))
    ensemble_r2 = r2_score(labels, y_pred_ensemble)
    ensemble_mae = np.mean(np.abs(labels - y_pred_ensemble))

    print("Ensemble Model Results:")
    print(f"RMSE = {ensemble_rmse:.4f}, R² = {ensemble_r2:.4f}, MAE = {ensemble_mae:.4f}")

    # 打印模型权重
    print("Model Weights:")
    for name, weight in weights.items():
        print(f"{name}: {weight:.4f}")

    # 绘制预测图
    plot_predictions(labels, y_pred_ensemble, 'Ensemble Model',
                     os.path.join(results_dir, 'ensemble_predictions.png'))

    # 创建一个所有预测结果的对比表格
    results_df = pd.DataFrame({
        'True': labels,
        f'Trad_{best_traditional_model}': y_pred_trad,
        f'CNN_{best_cnn_model}': y_pred_cnn,
        f'Comb_{best_combined_model}': y_pred_combined,
        'Ensemble': y_pred_ensemble
    })

    # 保存预测结果
    results_df.to_csv(os.path.join(results_dir, 'all_predictions.csv'), index=False)

    # 绘制所有模型的比较图
    methods = [f'Trad_{best_traditional_model}', f'CNN_{best_cnn_model}',
               f'Comb_{best_combined_model}', 'Ensemble']
    rmse_values = [
        np.sqrt(mean_squared_error(labels, y_pred_trad)),
        np.sqrt(mean_squared_error(labels, y_pred_cnn)),
        np.sqrt(mean_squared_error(labels, y_pred_combined)),
        ensemble_rmse
    ]
    r2_values = [
        r2_score(labels, y_pred_trad),
        r2_score(labels, y_pred_cnn),
        r2_score(labels, y_pred_combined),
        ensemble_r2
    ]

    # 创建比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # RMSE图
    ax1.bar(methods, rmse_values, color=['blue', 'green', 'red', 'purple'])
    ax1.set_title('RMSE Comparison')
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.set_xticklabels(methods, rotation=45, ha='right')

    # R²图
    ax2.bar(methods, r2_values, color=['blue', 'green', 'red', 'purple'])
    ax2.set_title('R² Comparison')
    ax2.set_ylabel('R² (higher is better)')
    ax2.set_xticklabels(methods, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存所有结果的摘要
    summary = {
        'dataset_size': len(labels),
        'label_statistics': {
            'mean': float(labels.mean()),
            'std': float(labels.std()),
            'min': float(labels.min()),
            'max': float(labels.max())
        },
        'traditional_features': {
            'shape': X_traditional.shape,
            'best_model': best_traditional_model,
            'metrics': traditional_results[best_traditional_model]
        },
        'cnn_features': {
            'shape': X_cnn.shape,
            'best_model': best_cnn_model,
            'metrics': cnn_results[best_cnn_model]
        },
        'combined_features': {
            'shape': X_combined.shape,
            'best_model': best_combined_model,
            'metrics': combined_results[best_combined_model]
        },
        'ensemble_model': {
            'weights': weights,
            'metrics': {
                'rmse': float(ensemble_rmse),
                'r2': float(ensemble_r2),
                'mae': float(ensemble_mae)
            }
        }
    }

    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAll results and visualizations saved to {results_dir}")

    # 返回最佳模型和特征提取器，以便后续使用
    if ensemble_r2 > max(r2_score(labels, y_pred_trad),
                         r2_score(labels, y_pred_cnn),
                         r2_score(labels, y_pred_combined)):
        print("\n==== 最佳模型：集成模型 ====")
        return {
            'type': 'ensemble',
            'weights': weights,
            'models': {
                f'Trad_{best_traditional_model}': traditional_models[best_traditional_model],
                f'CNN_{best_cnn_model}': traditional_models[best_cnn_model],
                f'Comb_{best_combined_model}': traditional_models[best_combined_model]
            },
            'scalers': {
                'traditional': scaler_trad,
                'cnn': scaler_cnn
            },
            'feature_extractors': {
                'cnn': extractor
            }
        }
    elif r2_score(labels, y_pred_combined) > max(r2_score(labels, y_pred_trad),
                                                 r2_score(labels, y_pred_cnn)):
        print(f"\n==== 最佳模型：组合特征 {best_combined_model} ====")
        traditional_models[best_combined_model].fit(X_combined, labels)
        return {
            'type': 'combined',
            'model': traditional_models[best_combined_model],
            'scalers': {
                'traditional': scaler_trad,
                'cnn': scaler_cnn
            },
            'feature_extractors': {
                'cnn': extractor
            }
        }
    elif r2_score(labels, y_pred_cnn) > r2_score(labels, y_pred_trad):
        print(f"\n==== 最佳模型：CNN特征 {best_cnn_model} ====")
        traditional_models[best_cnn_model].fit(X_cnn_scaled, labels)
        return {
            'type': 'cnn',
            'model': traditional_models[best_cnn_model],
            'scaler': scaler_cnn,
            'feature_extractor': extractor
        }
    else:
        print(f"\n==== 最佳模型：传统特征 {best_traditional_model} ====")
        traditional_models[best_traditional_model].fit(X_traditional_scaled, labels)
        return {
            'type': 'traditional',
            'model': traditional_models[best_traditional_model],
            'scaler': scaler_trad
        }

# 预测新样本的函数
def predict(best_model_info, image_path):
    """使用训练好的模型预测新样本"""
    try:
        if best_model_info['type'] == 'traditional':
            # 提取传统特征
            features = extract_combined_texture_features(image_path)
            # 标准化特征
            features_scaled = best_model_info['scaler'].transform([features])
            # 预测
            prediction = best_model_info['model'].predict(features_scaled)[0]

        elif best_model_info['type'] == 'cnn':
            # 提取CNN特征
            features = best_model_info['feature_extractor'].extract_features(image_path)
            # 标准化特征
            features_scaled = best_model_info['scaler'].transform([features])
            # 预测
            prediction = best_model_info['model'].predict(features_scaled)[0]

        elif best_model_info['type'] == 'combined':
            # 提取传统特征
            trad_features = extract_combined_texture_features(image_path)
            trad_features_scaled = best_model_info['scalers']['traditional'].transform([trad_features])

            # 提取CNN特征
            cnn_features = best_model_info['feature_extractors']['cnn'].extract_features(image_path)
            cnn_features_scaled = best_model_info['scalers']['cnn'].transform([cnn_features])

            # 组合特征
            combined_features = np.hstack((trad_features_scaled, cnn_features_scaled))

            # 预测
            prediction = best_model_info['model'].predict(combined_features)[0]

        elif best_model_info['type'] == 'ensemble':
            predictions = []

            # 提取传统特征
            trad_features = extract_combined_texture_features(image_path)
            trad_features_scaled = best_model_info['scalers']['traditional'].transform([trad_features])

            # 提取CNN特征
            cnn_features = best_model_info['feature_extractors']['cnn'].extract_features(image_path)
            cnn_features_scaled = best_model_info['scalers']['cnn'].transform([cnn_features])

            # 组合特征
            combined_features = np.hstack((trad_features_scaled, cnn_features_scaled))

            # 获取各个模型的预测
            for name, model in best_model_info['models'].items():
                if name.startswith('Trad_'):
                    pred = model.predict(trad_features_scaled)[0]
                elif name.startswith('CNN_'):
                    pred = model.predict(cnn_features_scaled)[0]
                elif name.startswith('Comb_'):
                    pred = model.predict(combined_features)[0]

                predictions.append((name, pred))

            # 加权平均
            weighted_sum = 0
            for (name, pred) in predictions:
                weighted_sum += pred * best_model_info['weights'][name]

            prediction = weighted_sum

        return prediction

    except Exception as e:
        print(f"预测时出错: {e}")
        return None

if __name__ == "__main__":
    best_model_info = main()

    # 示例：如何使用训练好的模型预测新样本
    # new_image_path = '/path/to/new/image.jpg'
    # prediction = predict(best_model_info, new_image_path)
    # print(f"Predicted value: {prediction:.4f}")