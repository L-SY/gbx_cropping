import os
import re

folder_path = ''  # 当前文件夹，您可以修改为您的图片所在文件夹路径

# 获取所有文件
all_files = os.listdir(folder_path)

# 使用正则表达式筛选符合 raw_数字.jpg 格式的文件
pattern = re.compile(r'temp_raw_(\d+)\.jpg')
files = []

for file in all_files:
    match = pattern.match(file)
    if match:
        index = int(match.group(1))
        files.append((file, index))

# 按索引排序
files.sort(key=lambda x: x[1], reverse=True)

# 从大到小重命名
for file, index in files:
    old_path = os.path.join(folder_path, file)
    new_index = index + 1
    new_name = f'raw_{new_index}.jpg'
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f'重命名: {file} -> {new_name}')

print(f'完成重命名 {len(files)} 个文件！')