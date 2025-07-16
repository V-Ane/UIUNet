import os
import shutil
import random

# 源文件夹路径
img_dir = "img"
label_dir = "label"

# 目标文件夹路径（自动创建）
train_img_dir = os.path.join("train_data", "img")
train_label_dir = os.path.join("train_data", "label")
test_img_dir = os.path.join("test_data", "img")
test_label_dir = os.path.join("test_data", "label")

# 创建目标文件夹（若不存在）
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 获取所有图片文件名
img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
# 排除可能的隐藏文件（如.DS_Store）
img_files = [f for f in img_files if not f.startswith('.')]

# 随机打乱文件顺序
random.shuffle(img_files)

# 计算划分比例（7:3）
total = len(img_files)
train_size = int(total * 0.7)

# 划分训练集和测试集文件名
train_files = img_files[:train_size]
test_files = img_files[train_size:]

# 复制文件到训练集
for file in train_files:
    # 图片文件
    src_img = os.path.join(img_dir, file)
    dst_img = os.path.join(train_img_dir, file)
    shutil.copy(src_img, dst_img)

    # 对应标签文件（文件名与图片完全一致，后缀也为.png）
    label_file = file  # 直接使用相同文件名
    src_label = os.path.join(label_dir, label_file)
    dst_label = os.path.join(train_label_dir, label_file)
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)
    else:
        print(f"警告：标签文件 {src_label} 不存在，已跳过")

# 复制文件到测试集
for file in test_files:
    # 图片文件
    src_img = os.path.join(img_dir, file)
    dst_img = os.path.join(test_img_dir, file)
    shutil.copy(src_img, dst_img)

    # 对应标签文件
    label_file = file
    src_label = os.path.join(label_dir, label_file)
    dst_label = os.path.join(test_label_dir, label_file)
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)
    else:
        print(f"警告：标签文件 {src_label} 不存在，已跳过")

print(f"划分完成！训练集 {len(train_files)} 个，测试集 {len(test_files)} 个")