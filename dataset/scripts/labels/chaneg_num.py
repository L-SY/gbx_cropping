import os
import re
import argparse

def rename_files(folder_path, pattern_str, template_str, offset=1, dry_run=False):
    """
    重命名文件夹中符合特定模式的文件

    参数:
    folder_path: 文件夹路径
    pattern_str: 匹配文件名的正则表达式字符串
    template_str: 新文件名的模板，使用{}表示数字位置
    offset: 数字偏移量，默认为1
    dry_run: 如果为True，只显示将要进行的重命名但不实际执行
    """
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在!")
        return

    # 获取所有文件
    all_files = os.listdir(folder_path)

    # 使用正则表达式筛选符合模式的文件
    pattern = re.compile(pattern_str)
    files = []

    for file in all_files:
        match = pattern.match(file)
        if match:
            index = int(match.group(1))
            files.append((file, index))

    if not files:
        print(f"没有找到符合模式 '{pattern_str}' 的文件!")
        return

    # 按索引排序
    files.sort(key=lambda x: x[1], reverse=True)

    print(f"找到 {len(files)} 个符合模式的文件")

    # 重命名
    for file, index in files:
        old_path = os.path.join(folder_path, file)
        new_index = index + offset
        new_name = template_str.format(new_index)
        new_path = os.path.join(folder_path, new_name)

        # 检查是否会覆盖现有文件
        if os.path.exists(new_path):
            print(f"警告: 文件 '{new_name}' 已存在，跳过重命名 '{file}'")
            continue

        if dry_run:
            print(f'将重命名: {file} -> {new_name}')
        else:
            try:
                os.rename(old_path, new_path)
                print(f'已重命名: {file} -> {new_name}')
            except Exception as e:
                print(f"重命名 '{file}' 时发生错误: {e}")

    if dry_run:
        print(f'模拟重命名完成，共 {len(files)} 个文件（实际未重命名）')
    else:
        print(f'完成重命名 {len(files)} 个文件！')

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='批量重命名文件')
    parser.add_argument('--folder', type=str, default='.',
                        help='文件夹路径 (默认: 当前文件夹)')
    parser.add_argument('--pattern', type=str, default=r'temp_raw_(\d+)\.jpg',
                        help='匹配文件名的正则表达式 (默认: "temp_raw_(\d+)\.jpg")')
    parser.add_argument('--template', type=str, default='raw_{}.jpg',
                        help='新文件名模板，使用{}表示数字位置 (默认: "raw_{}.jpg")')
    parser.add_argument('--offset', type=int, default=1,
                        help='数字偏移量 (默认: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示将要进行的更改，不实际重命名文件')

    # 解析命令行参数
    args = parser.parse_args()

    # 执行重命名
    rename_files(args.folder, args.pattern, args.template, args.offset, args.dry_run)

if __name__ == "__main__":
    main()