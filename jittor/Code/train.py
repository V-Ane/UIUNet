import os
import numpy as np
import glob
import cv2
import jittor as jt
from jittor import Module, nn
from data_loader import Rescale, RescaleT, RandomCrop, ToTensor, ToTensorLab, SalObjDataset
import random
from UIUNet import UIUNET
import matplotlib.pyplot as plt
import csv


if __name__ == '__main__':
    jt.flags.use_cuda = jt.has_cuda

    # ------- 1. 定义损失函数 --------
    bce_loss = nn.BCELoss(size_average=True)

    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, AA=True):
        loss0 = bce_loss(d0, labels_v)
        loss1 = bce_loss(d1, labels_v)
        loss2 = bce_loss(d2, labels_v)
        loss3 = bce_loss(d3, labels_v)
        loss4 = bce_loss(d4, labels_v)
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        if AA:
            print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f" % (
                loss0.data.item(), loss1.data.item(), loss2.data.item(), 
                loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()
            ))
        return loss0, loss

    # ------- 2. 设置训练数据集目录 --------
    model_name = 'uiunet'
    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('img' + os.sep)
    tra_label_dir = os.path.join('label' + os.sep)

    image_ext = '.png'
    label_ext = '.png'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    # ====== 新增：创建模型保存目录 ======
    os.makedirs(model_dir, exist_ok=True)

    epoch_num = 1000
    batch_size_train = 4
    batch_size_val = 3
    save_frq = 10520  # 保存频率

    # 获取训练图像和标签路径
    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        tra_lbl_name_list.append(os.path.join(data_dir, tra_label_dir, base_name + label_ext))

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")
    print("---")
    train_num = len(tra_img_name_list)

    # 训练数据加载器
    train_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=jt.transform.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)
        ])
    )
    train_loader = train_dataset.set_attrs(
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )

    # ====== 新增：验证数据集设置 ======
    test_data_dir = os.path.join(os.getcwd(), 'test_data' + os.sep)
    test_image_dir = os.path.join(test_data_dir, 'img' + os.sep)
    test_label_dir = os.path.join(test_data_dir, 'label' + os.sep)

    test_img_name_list = glob.glob(test_image_dir + '*' + image_ext)
    test_lbl_name_list = []
    for img_path in test_img_name_list:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        test_lbl_name_list.append(os.path.join(test_label_dir, base_name + label_ext))

    print("---")
    print("test images: ", len(test_img_name_list))
    print("test labels: ", len(test_lbl_name_list))
    print("---")
    print("---")
    val_num = len(test_img_name_list)

    # ====== 新增：验证数据加载器 (无随机裁剪) ======
    val_dataset = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        transform=jt.transform.Compose([
            RescaleT(320),
            ToTensorLab(flag=0)
        ])
    )
    val_loader = val_dataset.set_attrs(
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    train_loss_history = []
    train_tar_loss_history = []
    val_loss_history = []
    val_tar_loss_history = []
    epochs_list = []

    # ------- 3. 定义模型 --------
    net = UIUNET(3, 1)

    # ------- 4. 定义优化器 --------
    optimizer = jt.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    print("---define optimizer...")
    # ------- 5. 训练过程 --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 10520
    for epoch in range(epoch_num):
        # 训练阶段
        net.train()
        
        # 初始化每个epoch的损失累加器——————————————————————————————————————————————————————————————————————————
        epoch_running_loss = 0.0
        epoch_running_tar_loss = 0.0
        epoch_batches = 0

        for i, data in enumerate(train_loader):
            ite_num += 1           
            ite_num4val = ite_num4val + 1
            
            inputs = data['image'].float32()
            labels = data['label'].float32()

            # 前向传播
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            if ite_num % 20 == 0 or ite_num <= 20:
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
            else:
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels,False)
            # 反向传播和优化
            optimizer.step(loss)
            

            # ------------------------------新添加------------------------- by Vane
            # 累加批次损失
            epoch_running_loss += loss.data.item()
            epoch_running_tar_loss += loss2.data.item()
            epoch_batches += 1
            # ------------------------------新添加------------------------- by Vane

            # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()
            
            # 清理中间变量
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            if ite_num % 20 == 0 or ite_num <= 20:
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            # 定期保存模型
            if ite_num % save_frq == 0:
                model_path = os.path.join(model_dir, f"{model_name}_iter_{ite_num}.pkl")
                net.save(model_path)
                print(f"模型已保存至: {model_path}")
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0


        # 计算epoch平均损失
        epoch_avg_loss = epoch_running_loss / epoch_batches
        epoch_avg_tar_loss = epoch_running_tar_loss / epoch_batches

        # 记录训练损失
        train_loss_history.append(epoch_avg_loss)
        train_tar_loss_history.append(epoch_avg_tar_loss)
        epochs_list.append(epoch + 1)
        
        
        net.eval()
        test_running_loss = 0.0
        test_running_tar_loss = 0.0
        test_iterations = 0
        with jt.no_grad():
            for i, data in enumerate(val_loader):
                test_iterations += 1
                inputs = data['image'].float32()
                labels = data['label'].float32()

                d0, d1, d2, d3, d4, d5, d6 = net(inputs)
                # ====== 新增：验证时不打印详细损失 ======
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels, AA=False)
            
                
                test_running_loss += loss.data.item()
                test_running_tar_loss += loss2.data.item()
                
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        test_loss = test_running_loss / test_iterations
        test_tar_loss = test_running_tar_loss / test_iterations
        print(f"[Epoch {epoch + 1}/{epoch_num}] Test loss: {test_loss:.3f}, Test target loss: {test_tar_loss:.3f}")

        # 记录验证损失
        val_loss_history.append(test_loss)
        val_tar_loss_history.append(test_tar_loss)

        # # 保存每个epoch的模型
        # model_path = os.path.join(model_dir, f"{model_name}_iter_{ite_num}.pkl")
        # net.save(model_path)
        # print(f"Epoch模型已保存至: {model_path}")

    print("训练完成!")
    # 训练结束后绘制损失曲线
    plt.figure(figsize=(12, 6))

    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_loss_history, 'b-', label='Training Loss')
    plt.plot(epochs_list, val_loss_history, 'r-', label='Validation Loss')
    plt.title('Total Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制目标损失
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_tar_loss_history, 'b-', label='Training Tar Loss')
    plt.plot(epochs_list, val_tar_loss_history, 'r-', label='Validation Tar Loss')
    plt.title('Target Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.tight_layout()
    loss_curve_path = os.path.join(model_dir, f'{model_name}_loss_curve.png')
    plt.savefig(loss_curve_path)
    print(f'Loss curve saved at: {loss_curve_path}')
    
    # 准备导出数据
    loss_data = {
        'Epoch': epochs_list,
        'Train Loss': train_loss_history,
        'Train Tar Loss': train_tar_loss_history,
        'Validation Loss': val_loss_history,
        'Validation Tar Loss': val_tar_loss_history
    }

    # 保存为CSV文件
    csv_path = os.path.join(model_dir, f'{model_name}_loss_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Tar_Loss', 'Validation_Loss', 'Validation_Tar_Loss'])
        # 写入数据
        for i in range(len(epochs_list)):
            writer.writerow([
                epochs_list[i],
                train_loss_history[i],
                train_tar_loss_history[i],
                val_loss_history[i],
                val_tar_loss_history[i]
            ])

    print(f'Loss data saved at: {csv_path}')