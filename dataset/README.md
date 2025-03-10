## DATASET

> 为了方便的管理/拓展数据集所创建的包

##### 包结构:

```
dataset/
├── first_batch/
│   ├── fifteen/             # 存放专门十五倍参考倍率的裁剪后的图片
│   │   ├── images/          # 裁剪后的图片
│   │   └── labels.csv       # 对应的标签文件
│   ├── whole/               # 存放该批次中其它倍率的样本
│   │   ├── images/
│   │   └── labels.csv
│   └── raw_images/          # 存放从hk_camera中拍到的原始图片
│
├── second_batch/
│   ├── fifteen/
│   │   ├── images/
│   │   └── labels.csv
│   ├── whole/
│   │   ├── images/
│   │   └── labels.csv
│   └── raw_images/
│
├── xxxx_batch/              # 其他批次的数据
│   ├── fifteen/
│   ├── whole/
│   └── raw_images/
│
└── whole_fifteen/           # 合并后的十五倍数据集
    ├── images/              # 合并后的图片集合
    └── labels.csv           # 合并后的标签文件

scripts/
├── image_cropping/          # 用于从hk_camera话题中获取原始图片并裁切成正方形的脚本
│   ├── image_saver.py       # 从相机话题保存原始图像
│   └── image_cropper.py     # 裁剪图像为正方形
│
├── data_processing/         # 数据处理脚本
│   ├── labeling_tool.py     # 图像标注工具
│   └── data_augmentation.py # 数据增强工具
│
└── merge_dataset.py         # 用于合并两个数据集到指定路径的脚本
```

