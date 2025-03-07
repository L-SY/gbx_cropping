import pandas as pd
import sys

# 设置输入和输出文件路径
input_file = '/home/lsy/gbx_cropping_ws/src/image_tools/second_batch_of_samples/whole_second_batch_fifteen_samples/cropping/data/second_batch_of_samples.csv'  # 替换为您的输入文件路径
output_file = '/home/lsy/gbx_cropping_ws/src/image_tools/second_batch_of_samples/whole_second_batch_fifteen_samples/cropping/labels.csv'  # 替换为您的输出文件路径

try:
    # 读取输入CSV
    df = pd.read_csv(input_file)

    # 创建新数据
    output_data = []
    for _, row in df.iterrows():
        # 获取ID并创建文件名
        id_value = row['ID']
        if isinstance(id_value, float) and id_value.is_integer():
            id_value = int(id_value)

        filename = f"cropped_raw_{id_value}.jpg"

        # 获取值并添加到输出数据
        value = row['ComputedRate']
        output_data.append({
            'image_name': filename,
            'label': value
        })

    # 创建并保存输出CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    print(f"转换完成，已保存到：{output_file}")
    print(output_df.head())

except Exception as e:
    print(f"发生错误：{e}")
    sys.exit(1)