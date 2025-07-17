# 本项目是基于Jittor复现的 UIUNet模型

## 目录结构

项目根目录/ 

├── jittor/                  # jittor框架实现版本

│   ├── code/                # 核心代码文件

│   │   ├── 代码内容         # 具体代码（如train.py、UIUNet.py等）

│   │   └── saved_models/    # 训练保存的模型 

│   │       └── uiunet/      # 按模型名区分 

│   └── results/             # 实验结果（log、损失图、表格等） 

│ ├── pytorch/                 # pytorch框架实现版本 

│   ├── code/                # 核心代码文件

│   │   ├── 代码内容         # 具体代码（如train.py、UIUNet.py等）

│   │   └── saved_models/    # 训练保存的模型 

│   │       └── uiunet/      # 按模型名区分

│   └── results/             # 实验结果（log、损失图、表格等）

│ ├── 数据划分和使用.py          # 数据集划分相关内容 

│ └── README.md                # 项目说明（目录结构、运行方式等）

# 数据集

本次实验使用了文章中提到的两个数据集结合来达到一千张以上的图片防止快速达到过拟合

MSISTD(1076张) + SIRST(427张)



## 训练、测试命令

```py
#pytorch框架下
python train.py  #训练命令
python test.py   #测试命令

#jittor框架下
python train.py  #训练命令
python test.py   #测试命令
```

使用过程python文件均按照库中的目录结构，并保证train_data和test_data中具有img,labels一一对应

## loss curve

![image](https://github.com/V-Ane/UIUNet/blob/master/blogimg/Origin导出图_loss.png)

对应训练的log在对应其框架的Results中



## 实验配置

| **配置项**            | **参数值**                                  | **备注**           |
| :-------------------- | :------------------------------------------ | :----------------- |
| **训练批次**          | 1000                                        | epoch_num          |
| **Batch Size (训练)** | 4                                           | `batch_size_train` |
| **Batch Size (验证)** | 3                                           | `batch_size_val`   |
| **模型保存频率**      | 10520                                       | `save_frq`         |
| **数据加载线程数**    | 1                                           | num_worker         |
| **优化器**            | Adam                                        |                    |
| **学习率 (lr)**       | 0.001                                       |                    |
| **动量参数 (betas)**  | (0.9, 0.999)                                |                    |
| **GPU**               | 2080Ti                                      |                    |
| **CPU**               | 12 vCPU Intel Xeon Platinum 8255C @ 2.50GHz |                    |







## 备注

pytorch原开源代码中有些内容缺失（也可能不必要），但按照内容要求，我对其进行了略微修改与添加，如原代码训练中并未使用测试集进行计算测试集的损失，以及画图保存csv文件等













​																			

​																																																			   中北大学 隋航

​																																																							  Vane
