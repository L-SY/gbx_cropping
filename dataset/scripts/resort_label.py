import os
import pandas as pd
import csv
import argparse
import glob

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='根据图片顺序创建新的标签CSV文件')
    parser.add_argument('--path', '-p', type=str, required=True, help='包含图片和labels.csv的文件夹路径')
    parser.add_argument('--output', '-o', type=str, default='new_labels.csv', help='输出的新CSV文件名称 (默认: new_labels.csv)')
    parser.add_argument('--sort_by_size', '-s', action='store_true', help='按图片文件大小而不是文件名排序')

    # 解析命令行参数
    args = parser.parse_args()

    folder_path = args.path
    output_filename = args.output
    sort_by_size = args.sort_by_size

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    # 寻找名字包含label的CSV文件
    label_csv_files = glob.glob(os.path.join(folder_path, '*label*.csv'))

    if not label_csv_files:
        print(f"错误: 在 '{folder_path}' 中找不到包含'label'的CSV文件")
        return

    # 如果找到多个符合条件的文件，使用第一个
    labels_csv_path = label_csv_files[0]
    if len(label_csv_files) > 1:
        print(f"注意: 找到多个标签CSV文件，将使用: {os.path.basename(labels_csv_path)}")
        print(f"所有找到的标签文件: {[os.path.basename(f) for f in label_csv_files]}")
    else:
        print(f"使用标签文件: {os.path.basename(labels_csv_path)}")

    # 读取原始的labels.csv文件
    try:
        original_labels_df = pd.read_csv(labels_csv_path)
        print(f"成功读取标签文件: {labels_csv_path}")
        print(f"标签文件包含 {len(original_labels_df)} 条记录")
    except Exception as e:
        print(f"读取标签文件时出错: {str(e)}")
        return

    # 假设CSV文件中的第一列是图片名称，第二列是标签
    # 如果格式不同，请根据实际情况修改
    try:
        image_to_label = dict(zip(original_labels_df.iloc[:, 0], original_labels_df.iloc[:, 1]))
    except IndexError:
        print("错误: 标签文件格式不符合预期。请确保第一列为图片名称，第二列为标签值。")
        return

    # 获取文件夹中所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    image_files = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                image_files.append(file)

    if not image_files:
        print(f"警告: 在 '{folder_path}' 中找不到任何图片文件")
        return

    print(f"在文件夹中找到 {len(image_files)} 个图片文件")

    # 按照指定方式排序图片文件
    if sort_by_size:
        print("按照文件大小排序图片...")
        image_files.sort(key=lambda x: os.path.getsize(os.path.join(folder_path, x)))
    else:
        print("按照文件名排序图片...")
        image_files.sort()

    # 创建新的CSV文件，只包含按照文件夹中图片顺序排列的标签
    new_labels_path = os.path.join(folder_path, output_filename)

    try:
        with open(new_labels_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # 只写入标签值
            labels_found = 0
            for image_file in image_files:
                if image_file in image_to_label:
                    writer.writerow([image_to_label[image_file]])
                    labels_found += 1
                else:
                    print(f"警告: 在原始标签文件中找不到图片 '{image_file}' 的标签")

        print(f"已创建新的标签文件: {new_labels_path}")
        print(f"已写入 {labels_found} 个标签 (总共 {len(image_files)} 个图片)")

        if labels_found < len(image_files):
            print(f"警告: {len(image_files) - labels_found} 个图片没有对应的标签")

    except Exception as e:
        print(f"创建新标签文件时出错: {str(e)}")

if __name__ == "__main__":
    main()
