# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedRegressionLoss(nn.Module):
    """混合回归损失函数，加强对大误差的惩罚"""
    def __init__(self, mse_weight=0.5, l1_weight=0.3, huber_weight=0.2, error_amplification=1.0):
        super(MixedRegressionLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')  # 使用'none'以便自定义处理
        self.l1 = nn.L1Loss(reduction='none')
        self.huber = nn.SmoothL1Loss(reduction='none')

        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.huber_weight = huber_weight
        self.error_amplification = error_amplification  # 误差放大系数

    def forward(self, pred, target):
        # 计算基本损失
        mse_loss_raw = self.mse(pred, target)
        l1_loss_raw = self.l1(pred, target)
        huber_loss_raw = self.huber(pred, target)

        # 计算误差大小
        abs_error = torch.abs(pred - target)

        # 创建误差权重，对大误差给予更高的权重
        # 方法1：使用指数函数放大误差
        error_weights = torch.pow(abs_error, self.error_amplification)

        # 标准化权重使总和为batch_size，保持整体损失幅度一致
        error_weights = error_weights / torch.mean(error_weights) if torch.mean(error_weights) > 0 else error_weights

        # 应用权重到每个损失组件
        mse_loss = torch.mean(mse_loss_raw * error_weights)
        l1_loss = torch.mean(l1_loss_raw * error_weights)
        huber_loss = torch.mean(huber_loss_raw * error_weights)

        # 合并损失
        total_loss = (self.mse_weight * mse_loss +
                      self.l1_weight * l1_loss +
                      self.huber_weight * huber_loss)

        return total_loss


class FocalRegressionLoss(nn.Module):
    """Focal Loss变体用于回归，对大误差样本给予更高的关注"""
    def __init__(self, gamma=2.0, threshold=1.0):
        super(FocalRegressionLoss, self).__init__()
        self.gamma = gamma  # 调整因子，增加会提高大误差权重
        self.threshold = threshold  # 误差阈值，用于确定何时加强惩罚

    def forward(self, pred, target):
        # 计算绝对误差
        abs_error = torch.abs(pred - target)

        # 计算平方误差(MSE基础)
        squared_error = torch.pow(abs_error, 2)

        # 计算衰减因子，类似于分类Focal Loss中的(1-p)^gamma
        # 我们希望当误差大于阈值时，损失被放大
        decay_factor = torch.pow(abs_error / self.threshold, self.gamma)
        decay_factor = torch.where(abs_error > self.threshold, decay_factor, torch.ones_like(decay_factor))

        # 应用衰减因子到损失
        focal_loss = squared_error * decay_factor

        return torch.mean(focal_loss)


class AdaptiveHuberLoss(nn.Module):
    """自适应Huber损失，对大误差使用超线性惩罚"""
    def __init__(self, delta=1.0, beta=2.0):
        super(AdaptiveHuberLoss, self).__init__()
        self.delta = delta  # Huber损失的delta参数
        self.beta = beta    # 大误差区域的幂次，>1时会加强惩罚

    def forward(self, pred, target):
        abs_error = torch.abs(pred - target)

        # 标准Huber损失部分(对小误差)
        huber_small = 0.5 * torch.pow(abs_error, 2)

        # 对于大误差，使用超线性项而不是线性项
        huber_large = self.delta * torch.pow(abs_error, self.beta) / self.beta

        # 根据阈值选择应用哪种损失
        loss = torch.where(abs_error <= self.delta, huber_small, huber_large)

        return torch.mean(loss)


class LogCoshLoss(nn.Module):
    """Log-Cosh损失，类似Huber但更平滑，对大误差有强惩罚但不会过度增长"""
    def __init__(self, scale=1.0):
        super(LogCoshLoss, self).__init__()
        self.scale = scale  # 调整损失曲率的比例因子

    def forward(self, pred, target):
        error = (pred - target) * self.scale
        loss = torch.log(torch.cosh(error))
        return torch.mean(loss)


class AsymmetricLoss(nn.Module):
    """非对称损失函数，可以对正误差(高估)或负误差(低估)施加不同的惩罚"""
    def __init__(self, positive_weight=2.0, negative_weight=1.0):
        super(AsymmetricLoss, self).__init__()
        self.positive_weight = positive_weight  # 高估的惩罚权重
        self.negative_weight = negative_weight  # 低估的惩罚权重

    def forward(self, pred, target):
        error = pred - target

        # 对正误差(高估)和负误差(低估)应用不同权重
        weighted_error = torch.where(
            error >= 0,
            self.positive_weight * torch.pow(error, 2),
            self.negative_weight * torch.pow(error, 2)
        )

        return torch.mean(weighted_error)
