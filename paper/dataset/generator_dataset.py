import os
import argparse
from datasets import DatasetSplitter, DatasetAugmenter

def main():
    parser = argparse.ArgumentParser(description="Dataset preparation: splitting and augmentation for regression tasks")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the folder containing raw images and labels.csv')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed dataset')

    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)

    parser.add_argument('--train_aug_factor', type=int, default=5)
    parser.add_argument('--val_aug_factor', type=int, default=5)
    parser.add_argument('--test_aug_factor', type=int, default=5)

    parser.add_argument('--no_split', action='store_true', help='Skip dataset splitting')
    parser.add_argument('--no_augment', action='store_true', help='Skip dataset augmentation')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    labels_file = os.path.join(dataset_path, "labels.csv")

    split_dataset_path = os.path.join(output_path, 'split_dataset')
    augmented_dataset_path = os.path.join(output_path, 'augmented_dataset')

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(split_dataset_path, exist_ok=True)
    os.makedirs(augmented_dataset_path, exist_ok=True)

    if not os.path.exists(labels_file):
        raise ValueError(f"未找到标签文件: {labels_file}")
    print(f"找到标签文件: {labels_file}")

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

    if not args.no_augment:
        print("增强训练集...")
        train_augmenter = DatasetAugmenter(args.train_aug_factor, is_training=True)
        train_augmenter.augment_dataset(
            os.path.join(split_dataset_path, 'train'),
            os.path.join(augmented_dataset_path, 'train'),
            is_training=True
        )

        print("增强验证集...")
        val_augmenter = DatasetAugmenter(args.val_aug_factor, is_training=False)
        val_augmenter.augment_dataset(
            os.path.join(split_dataset_path, 'val'),
            os.path.join(augmented_dataset_path, 'val'),
            is_training=False
        )

        print("增强测试集...")
        test_augmenter = DatasetAugmenter(args.test_aug_factor, is_training=False)
        test_augmenter.augment_dataset(
            os.path.join(split_dataset_path, 'test'),
            os.path.join(augmented_dataset_path, 'test'),
            is_training=False
        )

    print(f"数据准备完成，输出保存至: {output_path}")

if __name__ == "__main__":
    main()
