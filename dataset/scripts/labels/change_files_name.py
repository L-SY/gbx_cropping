#!/usr/bin/env python3

import os
import argparse
import re

def rename_files(directory, prefix, pattern=None, dry_run=False):
    '''批量重命名文件夹中的图片文件

    支持两种模式:
    1. 从 1.jpg, 2.jpg... 改为 prefix_1.jpg, prefix_2.jpg...
    2. 从 pattern_1.jpg, pattern_2.jpg... 改为 pattern_prefix_1.jpg, pattern_prefix_2.jpg...
    '''
    # 确保目录存在
    if not os.path.isdir(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        return False

    # 获取所有jpg文件
    files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]

    renamed_count = 0

    # 处理简单数字命名的文件 (1.jpg, 2.jpg...)
    if pattern is None or pattern == "":
        number_files = [f for f in files if re.match(r'^\d+\.jpg$', f.lower())]
        number_files.sort(key=lambda f: int(re.search(r'^(\d+)', f).group(1)))

        for filename in number_files:
            # 提取数字部分
            number = re.search(r'^(\d+)', filename).group(1)
            new_filename = f"{prefix}_{number}.jpg"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # 如果是演习模式，只打印而不实际重命名
            if dry_run:
                print(f"将重命名: {filename} -> {new_filename}")
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"已重命名: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"重命名 {filename} 时出错: {e}")

    # 处理带前缀的文件 (pattern_1.jpg, pattern_2.jpg...)
    if pattern:
        pattern_regex = re.escape(pattern) + r'_(\d+)\.jpg$'
        pattern_files = [f for f in files if re.match(pattern_regex, f.lower())]
        pattern_files.sort(key=lambda f: int(re.search(pattern_regex, f.lower()).group(1)))

        for filename in pattern_files:
            # 提取数字部分
            number = re.search(pattern_regex, filename.lower()).group(1)
            new_filename = f"{pattern}_{prefix}_{number}.jpg"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # 如果是演习模式，只打印而不实际重命名
            if dry_run:
                print(f"将重命名: {filename} -> {new_filename}")
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"已重命名: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"重命名 {filename} 时出错: {e}")

    print(f"{'将' if dry_run else '已'}重命名 {renamed_count} 个文件")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量重命名图片文件')
    parser.add_argument('--directory', help='包含图片的目录路径')
    parser.add_argument('--prefix', default='back', help='文件名前缀 (默认: back)')
    parser.add_argument('--pattern', help='文件名匹配模式，例如 "cropped"')
    parser.add_argument('--dry-run', action='store_true', help='演习模式，只显示将要进行的更改而不实际执行')

    args = parser.parse_args()
    rename_files(args.directory, args.prefix, args.pattern, args.dry_run)
