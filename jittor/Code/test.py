import glob
import os

import numpy as np
from PIL import Image
from skimage import io
import jittor as jt

from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab
from UIUNet import UIUNET
# import torch.optim as optim
from metrics import *

def normPRED(d):
    ma = jt.max(d)
    mi = jt.min(d)
    diff = ma - mi
    # 处理常值张量情况
    if diff < 1e-8:
        return jt.zeros_like(d)
    return (d - mi) / diff


# 保存输出结果
def save_output(image_name, pred, d_dir):
    predict = pred  # 预测结果
    predict = predict.squeeze()  # 去除维度为1的维度
    predict_np = predict.numpy()  # 转换为NumPy数组

    # 将预测结果转换为PIL图像(0-255范围)
    im = Image.fromarray(predict_np * 255).convert('RGB')

    # 获取图像文件名
    img_name = image_name.split(os.sep)[-1]

    # 读取原始图像以获取尺寸
    image = io.imread(image_name)

    # 将预测结果缩放到原始图像尺寸
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    # 转换为NumPy数组
    pb_np = np.array(imo)

    # 处理文件名(去除扩展名)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    # 保存预测结果
    imo.save(d_dir + imidx + '.png')

def main():
    model_name = 'uiunet'

    # 构建测试数据路径
    image_dir = os.path.join(os.getcwd(), 'test_data', 'img')
    label_dir = os.path.join(os.getcwd(), 'test_data', 'label')

    # 模型文件路径
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name,  'uiunet_iter_10520.pkl')

    # 获取所有测试图像路径列表
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(f"找到 {len(img_name_list)} 张测试图像")

    # 获取所有标签图像路径列表
    label_name_list = glob.glob(label_dir + os.sep + '*')


    test_salobj_dataloader =SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=label_name_list,
        transform=jt.transform.Compose([
            RescaleT(320),  # 缩放图像到320x320
            ToTensorLab(flag=0)  # 转换为张量(flag=0表示不进行归一化)
        ]),
        batch_size=1,  # 批大小为1(每次处理一张图像)
        shuffle=False,  # 不随机打乱数据
        num_workers=1  # 使用1个子进程加载数据
    )


    # 创建UIU-NET模型(输入3通道RGB, 输出1通道分割图)
    net = UIUNET(3, 1)
    net.eval()
    net.load_state_dict(jt.load(model_dir))

    # 初始化评估指标
    iou_metric = SigmoidMetric()  # 整体IoU指标
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.55)  # 样本级IoU指标

    # 重置指标计数器
    iou_metric.reset()
    nIoU_metric.reset()

    # 初始化最佳指标值
    best_iou = 0
    best_nIoU = 0

    # 初始化总指标值
    total_iou = 0
    total_niou = 0

    # 遍历测试数据集
    for i_test, data_test in enumerate(test_salobj_dataloader):
        # 打印当前处理的图像名称
        print("推理中:", img_name_list[i_test].split(os.sep)[-1])

        # 获取输入图像
        inputs_test = data_test['image'].float32()
        # inputs_test = inputs_test.type(torch.FloatTensor)  # 转换为浮点张量

        # --------- 模型推理 ---------
        # 运行模型(UIU-NET有7个输出)
        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # 归一化第一个输出(主要预测结果)
        pred = d1[:, 0, :, :]  # 取第一个输出通道
        pred = normPRED(pred)  # 归一化到[0,1]

        # --------- 评估指标计算 ---------
        labels = data_test['label']  # 获取标签并移到CPU
        output = pred.unsqueeze(0)  # 添加批次维度并移到CPU

        # 更新指标
        iou_metric.update(output, labels)  # 更新整体IoU指标
        nIoU_metric.update(output, labels)  # 更新样本级IoU指标

        # 获取当前指标值
        _, IoU = iou_metric.get()
        _, nIoU = nIoU_metric.get()

        # --------- 更新最佳指标 ---------
        if IoU > best_iou:
            best_iou = IoU  # 更新最佳整体IoU

        if nIoU > best_nIoU:
            best_nIoU = nIoU  # 更新最佳样本级IoU

        # 累加指标值
        total_iou += IoU
        total_niou += nIoU

        # 释放中间变量(节省内存)
        del d1, d2, d3, d4, d5, d6, d7

    # --------- 计算平均指标 ---------
    # 计算平均IoU(总IoU除以样本数)
    avg_iou = total_iou / len(test_salobj_dataloader)
    avg_niou = total_niou / len(test_salobj_dataloader)

    # 打印结果
    print(f"平均IoU: {avg_iou}, 平均样本级IoU: {avg_niou:}")
    print(f"最佳IoU: {best_iou}, 最佳样本级IoU: {best_nIoU}")
# 程序入口
if __name__ == "__main__":
    jt.flags.use_cuda = 1
    main()