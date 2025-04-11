# train.py
import os
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms

# 导入自定义模块
from models import FrozenCNNRegressor
from datasets import RegressionDataset, DatasetSplitter, DatasetAugmenter
from trainer import Trainer
from utils import parse_args, set_seeds, save_config, generate_inference_code
from visualization import visualize_preprocessing

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置随机种子
    set_seeds(args.seed)

    # 设置设备
    import torch
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    print(f"使用设备: {device}")

    # 设置路径
    dataset_path = args.dataset_path
    output_path = args.output_path
    labels_file = os.path.join(dataset_path, "labels.csv")

    # 创建输出目录结构
    split_dataset_path = os.path.join(output_path, 'split_dataset')
    augmented_dataset_path = os.path.join(output_path, 'augmented_dataset')
    checkpoints_path = os.path.join(output_path, 'checkpoints')

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(split_dataset_path, exist_ok=True)
    os.makedirs(augmented_dataset_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # 检查标签文件是否存在
    if not os.path.exists(labels_file):
        raise ValueError(f"未找到标签文件: {labels_file}")

    print(f"找到标签文件: {labels_file}")

    # 创建实验文件夹（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(checkpoints_path, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # 分割数据集
    if not args.no_split:
        print("将数据集分割为训练、验证和测试集...")
        splitter = DatasetSplitter(
            source_dir=dataset_path,
            labels_file=labels_file,
            target_dir=split_dataset_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        splitter.split_dataset()

    # 数据增强
    if not args.no_augment:
        # 增强训练集
        print("增强训练集...")
        train_augmenter = DatasetAugmenter(augmentation_factor=args.train_aug_factor, is_training=True)
        train_augmenter.augment_dataset(
            os.path.join(split_dataset_path, 'train'),
            os.path.join(augmented_dataset_path, 'train'),
            is_training=True
        )

        # 增强验证集
        print("增强验证集...")
        val_augmenter = DatasetAugmenter(augmentation_factor=args.val_aug_factor, is_training=False)
        val_augmenter.augment_dataset(
            os.path.join(split_dataset_path, 'val'),
            os.path.join(augmented_dataset_path, 'val'),
            is_training=False
        )

        # 增强测试集
        print("增强测试集...")
        test_augmenter = DatasetAugmenter(augmentation_factor=args.test_aug_factor, is_training=False)
        test_augmenter.augment_dataset(
            os.path.join(split_dataset_path, 'test'),
            os.path.join(augmented_dataset_path, 'test'),
            is_training=False
        )

    # 可视化预处理步骤
    # 查找第一张图片用于可视化
    df = pd.read_csv(labels_file)
    if len(df) > 0:
        first_image = df.iloc[0]['image_name']
        sample_img_path = os.path.join(dataset_path, first_image)
        if os.path.exists(sample_img_path):
            visualize_preprocessing(
                sample_img_path,
                os.path.join(experiment_dir, 'preprocessing_visualization')
            )
        else:
            print(f"警告: 在 {sample_img_path} 找不到第一张图像")
    else:
        print("在标签文件中找不到图像")

    # 如果不训练，则退出
    if args.no_train:
        print("按要求跳过训练.")

        # 保存配置信息
        config = {
            'backbone': args.backbone,
            'train_aug_factor': args.train_aug_factor,
            'val_aug_factor': args.val_aug_factor,
            'test_aug_factor': args.test_aug_factor,
            'batch_size': args.batch_size,
            'initial_lr': args.learning_rate,
            'num_epochs': args.num_epochs,
            'unfreeze_at_epoch': args.unfreeze_epoch,
            'experiment_timestamp': timestamp,
            'model_type': 'FrozenCNN_with_FC',
            'dropout_rate': args.dropout_rate,
            'initial_value': args.initial_value,
            'no_train': args.no_train
        }

        save_config(config, os.path.join(experiment_dir, 'config.json'))
        return

    # 设置训练数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载增强数据集
    train_dataset = RegressionDataset(
        os.path.join(augmented_dataset_path, 'train'),
        transform=transform
    )
    val_dataset = RegressionDataset(
        os.path.join(augmented_dataset_path, 'val'),
        transform=transform
    )
    test_dataset = RegressionDataset(
        os.path.join(augmented_dataset_path, 'test'),
        transform=transform
    )

    print(f"数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 创建冻结CNN+FC模型
    model = FrozenCNNRegressor(
        backbone=args.backbone,
        pretrained=True,
        initial_value=args.initial_value,
        dropout_rate=args.dropout_rate
    )

    # 验证哪些层被冻结
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"使用 {args.backbone} 骨干网络创建冻结CNN+FC模型")
    print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({trainable_params/total_params:.2%})")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        save_dir=experiment_dir
    )

    # 设置早停耐心值
    trainer.patience = args.patience

    # 训练模型
    trainer.train(
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        # 如果需要解冻策略，使用正确的参数
        unfreeze_schedule = [
            (50, 2, 0.1),    # 在第80轮解冻最后2层
            (80, 4, 0.05),  # 在第120轮解冻最后4层
            (120, 8, 0.01)   # 在第160轮解冻最后8层
        ]
    )

    # 评估模型
    if not args.no_eval:
        trainer.evaluate_all_datasets()

    # 保存配置信息
    config = {
        'backbone': args.backbone,
        'train_aug_factor': args.train_aug_factor,
        'val_aug_factor': args.val_aug_factor,
        'test_aug_factor': args.test_aug_factor,
        'batch_size': args.batch_size,
        'initial_lr': args.learning_rate,
        'num_epochs': args.num_epochs,
        'unfreeze_at_epoch': args.unfreeze_epoch,
        'experiment_timestamp': timestamp,
        'model_type': 'FrozenCNN_with_FC',
        'dropout_rate': args.dropout_rate,
        'initial_value': args.initial_value,
        'best_epoch': trainer.history['best_epoch'],
        'best_val_loss': float(trainer.history['best_val_loss'])
    }

    save_config(config, os.path.join(experiment_dir, 'config.json'))

    # 创建推理示例代码
    generate_inference_code(args.backbone, experiment_dir)

    print(f"训练完成! 结果已保存到 {experiment_dir}")
    print("提示: 使用保存的 'best_model.pth' 文件进行推理，示例代码已保存到 inference_example.py")

if __name__ == "__main__":
    main()