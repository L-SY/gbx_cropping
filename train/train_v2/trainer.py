# trainer.py
import os
import json
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

class Trainer:
    """训练器类"""
    def __init__(self, model, train_loader, val_loader, test_loader=None, device='cuda',
                 learning_rate=0.001, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 定义混合损失函数
        from losses import MixedRegressionLoss, FocalRegressionLoss
        self.criterion = FocalRegressionLoss()

        # 只获取requires_grad=True的参数进行优化
        # 这将只更新未冻结的参数，大大提高训练效率
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        print(f"可训练参数数量: {len(params_to_update)}")

        # 使用AdamW优化器，带L2正则化
        self.optimizer = optim.AdamW(params_to_update, lr=learning_rate, weight_decay=1e-4)

        # 学习率调度器 - 带热重启的余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,        # 重启周期
            T_mult=2,      # 每次重启后的周期乘数
            eta_min=1e-6   # 最小学习率
        )

        # 记录训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

        # 早停计数器
        self.patience = 40
        self.patience_counter = 0
        self.early_stop = False

    def train_epoch(self):
        """训练一个轮次"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='Training')

        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()

            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self, data_loader, desc='Validating'):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_values = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc=desc):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(labels.cpu().numpy())

        return (total_loss / len(data_loader),
                np.array(predictions),
                np.array(true_values))

    def evaluate_all_datasets(self):
        """评估所有数据集并生成综合报告"""
        print("\n评估所有数据集...")
        # 加载最佳模型
        self.load_checkpoint('best_model.pth')

        # 创建结果存储字典
        results = {}

        # 评估训练集
        train_loss, train_preds, train_true = self.validate(self.train_loader, desc='Evaluating Training Set')
        results['train'] = {
            'loss': train_loss,
            'predictions': train_preds,
            'true_values': train_true,
            'r2': self.calculate_r2(train_preds, train_true),
            'rmse': np.sqrt(np.mean((train_preds - train_true) ** 2)),
            'mae': np.mean(np.abs(train_preds - train_true))
        }

        # 评估验证集
        val_loss, val_preds, val_true = self.validate(self.val_loader, desc='Evaluating Validation Set')
        results['val'] = {
            'loss': val_loss,
            'predictions': val_preds,
            'true_values': val_true,
            'r2': self.calculate_r2(val_preds, val_true),
            'rmse': np.sqrt(np.mean((val_preds - val_true) ** 2)),
            'mae': np.mean(np.abs(val_preds - val_true))
        }

        # 评估测试集
        test_loss, test_preds, test_true = self.validate(self.test_loader, desc='Evaluating Test Set')
        results['test'] = {
            'loss': test_loss,
            'predictions': test_preds,
            'true_values': test_true,
            'r2': self.calculate_r2(test_preds, test_true),
            'rmse': np.sqrt(np.mean((test_preds - test_true) ** 2)),
            'mae': np.mean(np.abs(test_preds - test_true))
        }

        # 保存结果
        self.plot_all_datasets_comparison(results)
        self.create_metrics_table(results)

        # 分析验证性能
        self.analyze_validation_performance(results)

        return results

    def analyze_validation_performance(self, results):
        """分析验证集性能以识别潜在问题"""
        print("\n分析验证集性能...")

        # 比较分布
        train_mean = np.mean(results['train']['true_values'])
        train_std = np.std(results['train']['true_values'])
        val_mean = np.mean(results['val']['true_values'])
        val_std = np.std(results['val']['true_values'])
        test_mean = np.mean(results['test']['true_values'])
        test_std = np.std(results['test']['true_values'])

        print(f"训练集: 均值={train_mean:.2f}, 标准差={train_std:.2f}")
        print(f"验证集: 均值={val_mean:.2f}, 标准差={val_std:.2f}")
        print(f"测试集: 均值={test_mean:.2f}, 标准差={test_std:.2f}")

        # 计算分布差异
        train_val_diff = abs(train_mean - val_mean) / train_std
        train_test_diff = abs(train_mean - test_mean) / train_std

        print(f"标准化均值差异 (训练-验证): {train_val_diff:.4f}")
        print(f"标准化均值差异 (训练-测试): {train_test_diff:.4f}")

        # 识别验证集中的异常值
        val_predictions = results['val']['predictions']
        val_true = results['val']['true_values']
        errors = np.abs(val_predictions - val_true)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        outlier_threshold = mean_error + 2 * std_error

        outliers = np.where(errors > outlier_threshold)[0]
        if len(outliers) > 0:
            print(f"在验证集中发现 {len(outliers)} 个潜在异常值")
            print(f"异常值阈值: {outlier_threshold:.4f}")
            print(f"5个最大误差:")
            top_errors = np.argsort(errors)[-5:]
            for idx in reversed(top_errors):
                print(f"  真实值: {val_true[idx]:.2f}, 预测值: {val_predictions[idx]:.2f}, 误差: {errors[idx]:.2f}")

        # 绘制分布比较
        plt.figure(figsize=(10, 6))
        plt.hist(results['train']['true_values'], alpha=0.5, bins=20, label='Training')
        plt.hist(results['val']['true_values'], alpha=0.5, bins=20, label='Validation')
        plt.hist(results['test']['true_values'], alpha=0.5, bins=20, label='Test')
        plt.title('Label Distribution Comparison')
        plt.xlabel('Label Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制误差分布
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20)
        plt.axvline(x=outlier_threshold, color='r', linestyle='--', label=f'Outlier threshold: {outlier_threshold:.2f}')
        plt.title('Validation Set Error Distribution')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'validation_error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_datasets_comparison(self, results):
        """绘制所有数据集的预测与真实值比较"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        datasets = ['train', 'val', 'test']
        titles = ['Training Set', 'Validation Set', 'Test Set']
        colors = ['blue', 'green', 'red']

        for i, (dataset, title, color) in enumerate(zip(datasets, titles, colors)):
            data = results[dataset]
            axes[i].scatter(data['true_values'], data['predictions'], alpha=0.6, s=30, c=color)

            # 添加完美预测线
            min_val = min(np.min(data['true_values']), np.min(data['predictions']))
            max_val = max(np.max(data['true_values']), np.max(data['predictions']))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

            # 添加回归线
            z = np.polyfit(data['true_values'], data['predictions'], 1)
            p = np.poly1d(z)
            axes[i].plot(data['true_values'], p(data['true_values']), 'g-', lw=1.5, alpha=0.7)

            # 设置标题和标签
            axes[i].set_title(f'{title}\nR² = {data["r2"]:.4f}, RMSE = {data["rmse"]:.4f}')
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].grid(True, alpha=0.3)

            # 设置坐标轴范围
            margin = (max_val - min_val) * 0.05  # 5% 边距
            axes[i].set_xlim(min_val - margin, max_val + margin)
            axes[i].set_ylim(min_val - margin, max_val + margin)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'all_datasets_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_metrics_table(self, results):
        """创建包含所有数据集性能指标的表格"""
        # 准备表格数据
        metrics_data = {
            'Dataset': ['Training Set', 'Validation Set', 'Test Set'],
            'Sample Size': [
                len(results['train']['true_values']),
                len(results['val']['true_values']),
                len(results['test']['true_values'])
            ],
            'Loss': [
                results['train']['loss'],
                results['val']['loss'],
                results['test']['loss']
            ],
            'R²': [
                results['train']['r2'],
                results['val']['r2'],
                results['test']['r2']
            ],
            'RMSE': [
                results['train']['rmse'],
                results['val']['rmse'],
                results['test']['rmse']
            ],
            'MAE': [
                results['train']['mae'],
                results['val']['mae'],
                results['test']['mae']
            ]
        }

        # 创建表格可视化
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')

        # 创建表格
        table = ax.table(
            cellText=[
                [f"{metrics_data['Dataset'][i]}",
                 f"{metrics_data['Sample Size'][i]}",
                 f"{metrics_data['Loss'][i]:.4f}",
                 f"{metrics_data['R²'][i]:.4f}",
                 f"{metrics_data['RMSE'][i]:.4f}",
                 f"{metrics_data['MAE'][i]:.4f}"]
                for i in range(3)
            ],
            colLabels=['Dataset', 'Sample Size', 'Loss', 'R²', 'RMSE', 'MAE'],
            loc='center',
            cellLoc='center'
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        # 设置表头样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 表头行
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')
            elif j == 0:  # 第一列
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#D9E1F2')
            elif i % 2 == 1:  # 奇数行
                cell.set_facecolor('#E9EDF4')

        plt.title('Model Performance Metrics Across Datasets', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_metrics_table.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 保存指标为CSV
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(self.save_dir, 'performance_metrics.csv'), index=False)

        # 保存为HTML表格
        html_table = metrics_df.to_html(index=False)
        with open(os.path.join(self.save_dir, 'performance_metrics.html'), 'w') as f:
            f.write("<html><head><style>")
            f.write("table {border-collapse: collapse; width: 100%; margin: 20px 0;}")
            f.write("th {background-color: #4472C4; color: white; font-weight: bold; text-align: center; padding: 10px;}")
            f.write("td {padding: 8px; text-align: center; border: 1px solid #ddd;}")
            f.write("tr:nth-child(even) {background-color: #E9EDF4;}")
            f.write("tr:hover {background-color: #ddd;}")
            f.write("</style></head><body>")
            f.write("<h2>Model Performance Metrics</h2>")
            f.write(html_table)
            f.write("</body></html>")

    # 在trainer.py中添加或修改以下方法

    def train(self, num_epochs, eval_every=1, unfreeze_schedule=None):
        """完整的训练过程，支持多阶段解冻策略

        参数:
            num_epochs (int): 总训练轮次
            eval_every (int): 每隔多少轮进行一次评估
            unfreeze_schedule (list): 解冻计划，格式为[(epoch, num_layers, lr_factor), ...]
                - epoch: 在哪个轮次开始解冻
                - num_layers: 解冻多少层
                - lr_factor: 学习率因子 (相对于初始学习率)
        """
        # 初始化解冻计划
        if unfreeze_schedule is None:
            unfreeze_schedule = []

        for epoch in range(num_epochs):
            if self.early_stop:
                print("触发早停!")
                break

            # 检查是否需要在当前轮次解冻层
            for schedule_epoch, num_layers, lr_factor in unfreeze_schedule:
                if epoch == schedule_epoch:
                    print(f"轮次 {epoch+1}: 解冻{num_layers}层进行微调")
                    if hasattr(self.model, 'unfreeze_layers'):
                        self.model.unfreeze_layers(num_layers)
                    elif hasattr(self.model, 'unfreeze_last_layers'):
                        self.model.unfreeze_last_layers(num_layers)
                    else:
                        print("模型未实现解冻方法")

                    # 调整学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.optimizer.defaults['lr'] * lr_factor
                    print(f"学习率调整为原来的 {lr_factor} 倍")
                    # 重新初始化优化器的学习率调度器
                    self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.optimizer,
                        T_0=20,
                        T_mult=2,
                        eta_min=1e-6
                    )
                    break  # 一轮只执行一次解冻

            print(f"\n轮次 {epoch+1}/{num_epochs}")
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.6f}")

            # 训练阶段
            train_loss = self.train_epoch()

            # 验证阶段 (基于eval_every参数)
            if (epoch + 1) % eval_every == 0:
                val_loss, predictions, true_values = self.validate(self.val_loader)

                # 更新学习率
                self.scheduler.step()

                # 记录历史
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rates'].append(current_lr)

                # 打印结果
                print(f"训练损失: {train_loss:.4f}")
                print(f"验证损失: {val_loss:.4f}")

                # 计算验证R^2和RMSE
                r2 = self.calculate_r2(predictions, true_values)
                rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
                print(f"验证 R²: {r2:.4f}, RMSE: {rmse:.4f}")

                # 保存最佳模型
                if val_loss < self.history['best_val_loss']:
                    self.history['best_val_loss'] = val_loss
                    self.history['best_epoch'] = epoch + 1
                    self.save_checkpoint(f'best_model.pth')
                    print(f"保存了新的最佳模型，验证损失: {val_loss:.4f}")
                    # 重置早停计数器
                    self.patience_counter = 0
                else:
                    # 增加早停计数器
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"在 {self.patience} 轮无改善后早停")
                        self.early_stop = True

                # 每5个评估周期保存检查点
                if (epoch + 1) % (eval_every * 5) == 0:
                    self.save_checkpoint(f'epoch_{epoch+1}.pth')

                # 绘制并保存当前预测结果
                self.plot_predictions(predictions, true_values, epoch+1)

                # 每10个评估周期绘制学习曲线
                if (epoch + 1) % (eval_every * 10) == 0:
                    self.plot_learning_curves()

        # 保存训练历史
        self.save_history()
        print(f"训练完成! 最佳验证损失: {self.history['best_val_loss']:.4f} 在轮次 {self.history['best_epoch']}")

        # 如果存在测试集，使用最佳模型评估它
        if self.test_loader:
            self.evaluate_test_set()

    def calculate_r2(self, predictions, true_values):
        """计算R² 决定系数"""
        mean_true = np.mean(true_values)
        ss_tot = np.sum((true_values - mean_true) ** 2)
        ss_res = np.sum((true_values - predictions) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # 添加小值以防止除零
        return r2

    def evaluate_test_set(self):
        """评估测试集性能"""
        print("\n在测试集上评估...")
        # 先加载最佳模型
        self.load_checkpoint('best_model.pth')

        # 评估测试集
        test_loss, test_preds, test_true = self.validate(self.test_loader, desc='Testing')

        # 计算测试集指标
        test_r2 = self.calculate_r2(test_preds, test_true)
        test_rmse = np.sqrt(np.mean((test_preds - test_true) ** 2))
        test_mae = np.mean(np.abs(test_preds - test_true))

        # 输出测试结果
        print(f"测试结果:")
        print(f"  损失: {test_loss:.4f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")

        # 绘制测试集预测
        self.plot_predictions(test_preds, test_true, epoch='test', save_path=os.path.join(self.save_dir, 'test_predictions.png'))

        # 保存测试结果
        test_results = {
            'loss': float(test_loss),
            'r2': float(test_r2),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'predictions': test_preds.tolist(),
            'true_values': test_true.tolist()
        }

        with open(os.path.join(self.save_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f)

    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 只有在存在且训练继续时才加载优化器和调度器
        if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hasattr(self, 'scheduler') and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        return checkpoint

    def plot_predictions(self, predictions, true_values, epoch, save_path=None):
        """绘制预测结果"""
        plt.figure(figsize=(10, 8))

        # 计算性能指标
        r2 = self.calculate_r2(predictions, true_values)
        rmse = np.sqrt(np.mean((predictions - true_values) ** 2))

        # 绘制散点图
        plt.scatter(true_values, predictions, alpha=0.6, s=40)

        # 添加完美预测线
        min_val = min(np.min(true_values), np.min(predictions))
        max_val = max(np.max(true_values), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        # 添加回归线
        z = np.polyfit(true_values, predictions, 1)
        p = np.poly1d(z)
        plt.plot(true_values, p(true_values), 'g-', lw=1.5, alpha=0.7)

        # 添加图表标题和坐标轴标签
        if isinstance(epoch, str):
            title = f'Predictions vs True Values - {epoch}'
        else:
            title = f'Predictions vs True Values - Epoch {epoch}'
        plt.title(f'{title}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid(True, alpha=0.3)

        # 设置坐标轴范围
        margin = (max_val - min_val) * 0.05  # 5% 边距
        plt.xlim(min_val - margin, max_val + margin)
        plt.ylim(min_val - margin, max_val + margin)

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_learning_curves(self):
        """绘制学习曲线"""
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(12, 10))

        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        plt.axvline(x=self.history['best_epoch'], color='g', linestyle='--', alpha=0.7)
        plt.text(self.history['best_epoch'], min(self.history['val_loss']),
                 f'Best: {self.history["best_val_loss"]:.4f}',
                 verticalalignment='bottom', horizontalalignment='right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 绘制学习率曲线
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.history['learning_rates'], 'g-')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.yscale('log')  # 对数刻度使学习率变化更容易看到
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_history(self):
        """保存训练历史"""
        history_file = os.path.join(self.save_dir, 'training_history.json')
        # 将NumPy数组转换为Python列表以进行JSON序列化
        serializable_history = {
            'train_loss': [float(x) for x in self.history['train_loss']],
            'val_loss': [float(x) for x in self.history['val_loss']],
            'learning_rates': [float(x) for x in self.history['learning_rates']],
            'best_val_loss': float(self.history['best_val_loss']),
            'best_epoch': self.history['best_epoch']
        }
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f)
