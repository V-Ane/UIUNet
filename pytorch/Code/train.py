
import random
import os

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
import cv2

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import UIUNET

import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告
# from lightning.pytorch import Trainer, seed_everything
#
# seed_everything(42, workers=True)
# trainer = Trainer(deterministic=True)

if __name__ == '__main__':
    seed = 7
    # ====== 新增：设置随机种子 ======
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
    
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False  # train speed is slower after enabling this opts.
    
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    
        # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
        # torch.use_deterministic_algorithms(True)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    set_seed(seed)  # 在程序开始时调用
    # ------- 1. define loss function --------

    bce_loss = nn.BCELoss(size_average=True)

    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v,AA = True):

        loss0 = bce_loss(d0, labels_v)
        loss1 = bce_loss(d1, labels_v)
        loss2 = bce_loss(d2, labels_v)
        loss3 = bce_loss(d3, labels_v)
        loss4 = bce_loss(d4, labels_v)
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        if AA:
            print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
            loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
            loss5.data.item(), loss6.data.item()))

        return loss0, loss

    # ------- 2. set the directory of training dataset --------

    model_name ='uiunet'

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('img' + os.sep)
    tra_label_dir = os.path.join('label' + os.sep)

    image_ext = '.png'
    label_ext = '.png'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    epoch_num = 1000
    batch_size_train = 4
    batch_size_val = 3
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")
    print("---")

    train_num = len(tra_img_name_list)
    g = torch.Generator()
    g.manual_seed(7)  # 为DataLoader设置固定种子

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0, drop_last=True,generator=g, worker_init_fn=worker_init_fn ) #shuffle=True 乱序
    # salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1,
    #                                drop_last=True)

    # ------------------------------新添加------------------------- by Vane
    # 设置测试数据集路径
    test_data_dir = os.path.join(os.getcwd(), 'test_data' + os.sep)
    test_image_dir = os.path.join(test_data_dir, 'img' + os.sep)
    test_label_dir = os.path.join(test_data_dir, 'label' + os.sep)

    # 获取测试图像和标签路径列表
    test_img_name_list = glob.glob(test_image_dir + '*' + image_ext)
    test_lbl_name_list = []
    for img_path in test_img_name_list:
        img_name = img_path.split(os.sep)[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        test_lbl_name_list.append(test_label_dir + imidx + label_ext)

    print("---")
    print("test images: ", len(test_img_name_list))
    print("test labels: ", len(test_lbl_name_list))
    print("---")
    print("---")

    val_num = len(test_img_name_list)
    g_test = torch.Generator()
    g_test.manual_seed(42)

    # 创建测试数据集和数据加载器
    test_dataset = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),  # 测试时不需要随机裁剪
            ToTensorLab(flag=0)]))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, num_workers=0,
                                 generator=g_test)

    # 创建用于存储损失值的列表
    train_loss_history = []
    train_tar_loss_history = []
    val_loss_history = []
    val_tar_loss_history = []
    epochs_list = []

    # ------------------------------新添加------------------------- by Vane
    # ------- 3. define model --------
    net = UIUNET(3, 1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 10520 # save the model every 2000 iterations

    for epoch in range(0, epoch_num):
        net.train()
        # ------------------------------新添加------------------------- by Vane
        # 初始化每个epoch的损失累加器
        epoch_running_loss = 0.0
        epoch_running_tar_loss = 0.0
        epoch_batches = 0
        # ------------------------------新添加------------------------- by Vane
        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            if ite_num % 20 == 0 or ite_num <= 20:
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            else:
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v,False)

            # ------------------------------新添加------------------------- by Vane
            # 累加批次损失
            epoch_running_loss += loss.data.item()
            epoch_running_tar_loss += loss2.data.item()
            epoch_batches += 1
            # ------------------------------新添加------------------------- by Vane

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss
            if ite_num % 20 == 0 or ite_num <= 20:
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:

                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0
        # ------------------------------新添加------------------------- by Vane
        # 计算epoch平均损失
        epoch_avg_loss = epoch_running_loss / epoch_batches
        epoch_avg_tar_loss = epoch_running_tar_loss / epoch_batches

        # 记录训练损失
        train_loss_history.append(epoch_avg_loss)
        train_tar_loss_history.append(epoch_avg_tar_loss)
        epochs_list.append(epoch + 1)


    #------------------------------新添加------------------------- by Vane
        # 测试阶段 - 每个epoch结束后进行一次测试
        net.eval()
        test_running_loss = 0.0
        test_running_tar_loss = 0.0
        test_iterations = 0

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                test_iterations += 1
                inputs, labels = data['image'], data['label']
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    inputs_v, labels_v = inputs.cuda(), labels.cuda()
                else:
                    inputs_v, labels_v = inputs, labels

                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v,False)

                test_running_loss += loss.data.item()
                test_running_tar_loss += loss2.data.item()

                del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        test_loss = test_running_loss / test_iterations
        test_tar_loss = test_running_tar_loss / test_iterations
        print(f"[Epoch {epoch + 1}/{epoch_num}] Test loss: {test_loss:.3f}, Test target loss: {test_tar_loss:.3f}")

        # 记录验证损失
        val_loss_history.append(test_loss)
        val_tar_loss_history.append(test_tar_loss)

        # 保存每个epoch的模型
        # torch.save(net.state_dict(), model_dir + f"{model_name}_epoch_{epoch + 1}_test_{test_loss:.3f}.pth")
        #------------------------------新添加------------------------- by Vane
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

    # 显示图像（如果环境支持）
    # plt.show()

    # 准备导出数据
    loss_data = {
        'Epoch': epochs_list,
        'Train_Loss': train_loss_history,
        'Train_Tar_Loss': train_tar_loss_history,
        'Validation_Loss': val_loss_history,
        'Validation_Tar_Loss': val_tar_loss_history
    }

    # 创建DataFrame
    loss_df = pd.DataFrame(loss_data)

    # # 保存为Excel文件（Origin可以直接读取）
    # excel_path = os.path.join(model_dir, f'{model_name}_loss_data.xlsx')
    # loss_df.to_excel(excel_path, index=False)

    # 也可以保存为CSV文件
    csv_path = os.path.join(model_dir, f'{model_name}_loss_data.csv')
    loss_df.to_csv(csv_path, index=False)

    print(f'Loss data saved at: {csv_path}')