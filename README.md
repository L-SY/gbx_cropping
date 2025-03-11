# gbx_cropping
Use hk_camera realize Image cropping and stitching

#### 训练代码使用

> dataset_path路径下包含图片和对应的labels.csv文件
>
> output_path下会输出增强数据集，checkpoint和分割后的数据集

```
python frozen_cnn_train.py --dataset_path /home/lsy/gbx_cropping_ws/src/dataset/dataset/first_batch/fifteen --output_path /home/lsy/gbx_cropping_ws/src/train/first_batch
```



#### 数据集完整排列

```
python whole_images_montage.py --dataset_path /home/lsy/gbx_cropping_ws/src/dataset/dataset/whole_fifteen --output_path /home/lsy/gbx_cropping_ws/src/dataset/scripts
```

