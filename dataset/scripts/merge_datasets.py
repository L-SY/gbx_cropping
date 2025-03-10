import os
import shutil
import pandas as pd
import argparse
import re

def extract_batch_name(path):
    """
    从路径中提取包含'batch'的部分
    例如从 '/home/lsy/gbx_cropping_ws/src/dataset/first_batch/fifteen' 提取 'first_batch'
    """
    path_parts = path.split(os.sep)
    for part in path_parts:
        if 'batch' in part.lower():
            return part
    return None

def merge_datasets(dataset1_path, dataset2_path, output_path):
    """
    合并两个数据集及其对应的标签文件到目标路径
    """
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 标签文件路径
    dataset1_label_path = os.path.join(dataset1_path, 'labels.csv')
    dataset2_label_path = os.path.join(dataset2_path, 'labels.csv')
    output_label_path = os.path.join(output_path, 'labels.csv')

    # 读取标签文件
    df1 = pd.read_csv(dataset1_label_path)
    df2 = pd.read_csv(dataset2_label_path)

    # 创建新的合并标签DataFrame
    merged_df = pd.DataFrame(columns=['image_name', 'label'])

    # 获取数据集批次名称
    dataset1_batch = extract_batch_name(dataset1_path)
    dataset2_batch = extract_batch_name(dataset2_path)

    dataset1_prefix = dataset1_batch if dataset1_batch else os.path.basename(os.path.normpath(dataset1_path))
    dataset2_prefix = dataset2_batch if dataset2_batch else os.path.basename(os.path.normpath(dataset2_path))

    # 处理第一个数据集
    print(f"处理第一个数据集: {dataset1_path} (前缀: {dataset1_prefix})")
    process_dataset(df1, dataset1_path, output_path, merged_df, dataset1_prefix)

    # 处理第二个数据集
    print(f"处理第二个数据集: {dataset2_path} (前缀: {dataset2_prefix})")
    process_dataset(df2, dataset2_path, output_path, merged_df, dataset2_prefix)

    # 保存合并后的标签文件
    merged_df.to_csv(output_label_path, index=False)
    print(f"合并完成! 共有 {len(merged_df)} 个文件被合并到 {output_path}")
    print(f"合并的标签文件已保存到 {output_label_path}")

def process_dataset(df, source_path, target_path, merged_df, dataset_prefix):
    """
    处理单个数据集，复制图片到目标路径并更新标签
    """
    # 检查目标路径是否与源路径相同
    same_path = os.path.abspath(source_path) == os.path.abspath(target_path)

    # 简化dataset_prefix，移除非字母数字字符
    clean_prefix = re.sub(r'[^a-zA-Z0-9]', '', dataset_prefix)

    for index, row in df.iterrows():
        original_image_name = row['image_name']
        label = row['label']

        # 检查原始文件名是否已经包含batch
        if 'batch' in original_image_name.lower():
            # 如果已经包含batch，保持原命名
            new_image_name = original_image_name
        else:
            # 获取原始文件的基本名称和扩展名
            base_name, ext = os.path.splitext(original_image_name)

            # 创建新的文件名，格式：前缀_原始名称.扩展名
            new_image_name = f"{clean_prefix}_{base_name}{ext}"

        # 源文件路径
        src_path = os.path.join(source_path, original_image_name)

        # 如果源路径和目标路径相同，我们需要先创建临时文件名
        if same_path:
            # 创建临时文件名以避免覆盖
            temp_image_name = f"temp_{new_image_name}"
            dst_path = os.path.join(target_path, temp_image_name)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                # 将临时文件名添加到合并标签DataFrame
                merged_df.loc[len(merged_df)] = {'image_name': temp_image_name, 'label': label}
                print(f"  临时复制: {original_image_name} -> {temp_image_name}")
            else:
                print(f"  警告: 找不到文件 {src_path}")
        else:
            # 目标文件路径
            dst_path = os.path.join(target_path, new_image_name)

            # 检查目标文件是否已存在，如果存在则添加递增的数字
            counter = 1
            original_new_name = new_image_name
            while os.path.exists(dst_path):
                name_parts = os.path.splitext(original_new_name)
                new_image_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                dst_path = os.path.join(target_path, new_image_name)
                counter += 1

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                # 添加到合并标签DataFrame
                merged_df.loc[len(merged_df)] = {'image_name': new_image_name, 'label': label}
                print(f"  复制: {original_image_name} -> {new_image_name}")
            else:
                print(f"  警告: 找不到文件 {src_path}")

def rename_temp_files(path):
    """重命名临时文件，移除'temp_'前缀"""
    label_path = os.path.join(path, 'labels.csv')
    if os.path.exists(label_path):
        df = pd.read_csv(label_path)
        new_names = []

        for index, row in df.iterrows():
            image_name = row['image_name']
            if image_name.startswith('temp_'):
                # 移除'temp_'前缀
                new_name = image_name[5:]
                src_path = os.path.join(path, image_name)
                dst_path = os.path.join(path, new_name)

                # 检查目标文件是否已存在，如果存在则添加递增的数字
                counter = 1
                original_new_name = new_name
                while os.path.exists(dst_path):
                    name_parts = os.path.splitext(original_new_name)
                    new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                    dst_path = os.path.join(path, new_name)
                    counter += 1

                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                    print(f"  重命名: {image_name} -> {new_name}")
                    new_names.append(new_name)
                else:
                    new_names.append(image_name)
                    print(f"  警告: 找不到临时文件 {src_path}")
            else:
                new_names.append(image_name)

        # 更新标签文件
        df['image_name'] = new_names
        df.to_csv(label_path, index=False)
        print(f"标签文件已更新，移除了临时文件前缀")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='合并两个数据集及其标签')
    parser.add_argument('--dataset_path1', required=True, help='第一个数据集的路径')
    parser.add_argument('--dataset_path2', required=True, help='第二个数据集的路径')
    parser.add_argument('--output_path', required=True, help='输出数据集的路径')

    # 解析命令行参数
    args = parser.parse_args()

    # 合并数据集
    merge_datasets(args.dataset_path1, args.dataset_path2, args.output_path)

    # 如果输出路径与任一输入路径相同，重命名临时文件
    if (os.path.abspath(args.output_path) == os.path.abspath(args.dataset_path1) or
            os.path.abspath(args.output_path) == os.path.abspath(args.dataset_path2)):
        rename_temp_files(args.output_path)