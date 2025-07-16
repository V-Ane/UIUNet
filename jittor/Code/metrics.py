import numpy as np
import jittor as jt

class SigmoidMetric():
    """
    用于计算二分类图像分割任务的评估指标
    包含像素精度(pixAcc)和平均交并比(mIoU)
    """


    def __init__(self):
        self.reset() #初始化计数器

    def update(self,pred,labels):
        """
        使用当前批次的预测结果和真实标签更新指标
        参数:
        pred: 模型预测的输出张量(形状为[batch, height, width])
        labels: 真实标签张量(形状与pred相同, 值应为0或1)
        """
        correct,labeled = self.batch_pix_accuracy(pred,labels) #当前批次的正确预测像素数和总标记像素数

        inter,union = self.batch_intersection_union(pred,labels) #当前批次的交集和并集


        #累加总计数器
        self.total_correct += correct   #总正确像素数
        self.total_label += labeled     #总标记像素数（前景）
        self.total_inter += inter      #总交集像素数
        self.total_union += union       #总并集像素数

    def get(self):
        """计算并返回最终评估指标"""
        small_value = np.spacing(1) #添加一个非常小的数防止除0

        # 像素精度   正确像素/总标记像素
        pixAcc = 1.0 * self.total_correct /(small_value + self.total_label)

        #交并比IoU
        IoU = 1.0 * self.total_inter / (small_value + self.total_union)

        #mIoU 二分类IoU只有一个值直接取平均
        mIoU = IoU.mean()

        #返回像素精度和mIoU
        return pixAcc, mIoU

    def reset(self):
        """重置所有计数器为零"""
        self.total_inter = 0    # 总交集像素数
        self.total_union = 0    # 总并集像素数
        self.total_correct = 0 # 总正确预测像素数
        self.total_label = 0    # 总标记像素数(前景像素)

    def batch_pix_accuracy(self, output, target):
        """
        计算当前批次的像素精度

        参数:
        output: 模型输出张量(未经过sigmoid)
        target: 真实标签张量

        返回:
        pixel_correct: 正确预测的像素数
        pixel_labeled: 标记的像素数(前景像素总数)
        """

        assert output.shape == target.shape
        output = output.numpy()
        target = target.numpy()

        # 使用阈值0.22将预测二值化(大于0.22变为1，否则为0)
        predict = (output > 0.22).astype('int64')

        # 计算被标记的前景像素总数
        pixel_labeled = np.sum(target > 0)

        # 计算正确预测的前景像素数  (target > 0))取前景正确的
        pixel_correct = np.sum((predict == target) * (target > 0))

        # 验证正确像素数不超过标记像素数(合理性检查)
        assert pixel_correct <= pixel_labeled

        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        """
        计算当前批次的交集和并集(用于IoU计算)

        参数:
        output: 模型输出张量
        target: 真实标签张量

        返回:
        area_inter: 交集像素数
        area_union: 并集像素数
        """
        # 设置直方图参数(因为是二分类，所以只有一个类别)
        mini = 1  # 类别最小值(前景类别为1)
        maxi = 1  # 类别最大值
        nbins = 1  # 直方图bin数量(等于类别数)

        # 使用阈值0.22二值化预测结果
        predict = (output.numpy() > 0.22).astype('int64')
        # 转换标签为int64类型
        target = target.numpy().astype('int64')

        # 计算交集: 预测正确的前景区域(TP)
        intersection = predict * (predict == target)

        # 使用直方图计算各区域的像素数:
        # np.histogram() 计算数组中值的分布
        # bins: 将值分成多少个区间
        # range: 值的范围(mini到maxi)

        # 计算交集区域的像素数
        # area_inter 是一个数组，包含每个区间的计数(这里只有一个区间)
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))

        # 计算预测区域的像素数(预测为前景的总像素数)
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))

        # 计算标签区域的像素数(真实前景的总像素数)
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

        # 计算并集: 预测区域 + 标签区域 - 交集区域
        area_union = area_pred + area_lab - area_inter

        # 验证交集面积不大于并集面积(合理性检查)
        assert (area_inter <= area_union).all()

        return area_inter, area_union


class SamplewiseSigmoidMetric():
    """
    按样本计算评估指标(每个样本单独计算IoU)
    """

    def __init__(self, nclass, score_thresh=0.5):
        """
        初始化
        nclass: 类别数(二分类为1)
        score_thresh: 二值化阈值(默认0.5)
        """
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()  # 初始化计数器

    def update(self, preds, labels):
        """使用当前批次的预测和标签更新指标"""
        # 计算当前批次每个样本的交集和并集
        inter_arr, union_arr = self.batch_intersection_union(
            preds, labels, self.nclass, self.score_thresh
        )
        # 将结果追加到总数组中
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """计算并返回每个样本的IoU和平均IoU"""
        # 计算每个样本的IoU = 交集 / 并集
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)

        # 计算平均IoU
        mIoU = IoU.mean()

        return IoU, mIoU  # 返回每个样本的IoU和平均IoU

    def reset(self):
        """重置计数器为空数组"""
        self.total_inter = np.array([])  # 存储每个样本的交集面积
        self.total_union = np.array([])  # 存储每个样本的并集面积
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target, nclass, score_thresh):
        """
        计算批次中每个样本的交集和并集

        参数:
        output: 模型输出张量
        target: 真实标签张量
        nclass: 类别数
        score_thresh: 二值化阈值

        返回:
        area_inter_arr: 每个样本的交集像素数
        area_union_arr: 每个样本的并集像素数
        """
        # 设置直方图参数
        mini = 1  # 类别最小值
        maxi = 1  # 类别最大值
        nbins = 1  # 直方图bin数

        # 应用sigmoid激活函数将输出转换为概率
        predict = (jt.sigmoid(output).numpy() > score_thresh).astype('int64')

        # 转换标签为int64类型
        target = target.numpy().astype('int64')

        # 计算交集: 预测正确的前景区域(TP)
        intersection = predict * (predict == target)

        # 获取批次大小(样本数量)
        num_sample = intersection.shape[0]

        # 初始化存储每个样本结果的数组
        area_inter_arr = np.zeros(num_sample)  # 交集像素数
        area_pred_arr = np.zeros(num_sample)  # 预测区域像素数
        area_lab_arr = np.zeros(num_sample)  # 标签区域像素数
        area_union_arr = np.zeros(num_sample)  # 并集像素数

        # 遍历批次中的每个样本
        for b in range(num_sample):
            # 计算当前样本的交集区域的像素数
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter  # 存储结果

            # 计算当前样本的预测区域的像素数
            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred  # 存储结果

            # 计算当前样本的标签区域的像素数
            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab  # 存储结果

            # 计算并集: 预测区域 + 标签区域 - 交集区域
            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union  # 存储结果

            # 验证交集面积不大于并集面积(合理性检查)
            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr


class ROCMetric():
    """
    用于计算ROC曲线(接收者操作特征曲线)的指标
    """

    def __init__(self, nclass, bins):
        """
        初始化
        nclass: 类别数
        bins: 将0-1区间划分的阈值数量
        """
        self.nclass = nclass
        self.bins = bins

        # 初始化存储每个阈值的计数数组
        self.tp_arr = np.zeros(self.bins + 1)  # 真正例数(TP)
        self.pos_arr = np.zeros(self.bins + 1)  # 实际正例数(TP + FN)
        self.fp_arr = np.zeros(self.bins + 1)  # 假正例数(FP)
        self.neg_arr = np.zeros(self.bins + 1)  # 实际负例数(TN + FP)

    def update(self, preds, labels):
        """使用当前批次的预测和标签更新所有阈值的指标"""
        # 遍历所有阈值(从0到1，步数为bins+1)
        for iBin in range(self.bins + 1):
            # 计算当前阈值(0到1之间均匀分布)
            # 例如: bins=10, 则阈值为0.0, 0.1, 0.2, ..., 1.0
            score_thresh = (iBin + 0.0) / self.bins

            # 计算当前阈值下的TP、实际正例、FP、实际负例
            i_tp, i_pos, i_fp, i_neg = cal_tp_pos_fp_neg(
                preds, labels, self.nclass, score_thresh
            )

            # 累加到总计数器中
            self.tp_arr[iBin] += i_tp  # 真正例
            self.pos_arr[iBin] += i_pos  # 实际正例
            self.fp_arr[iBin] += i_fp  # 假正例
            self.neg_arr[iBin] += i_neg  # 实际负例

    def get(self):
        """计算并返回真正率(TPR)和假正率(FPR)"""
        # 真正率(TPR) = TP / (TP + FN) = TP / 实际正例
        # 添加0.001防止除以零
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)

        # 假正率(FPR) = FP / (FP + TN) = FP / 实际负例
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        return tp_rates, fp_rates  # 返回所有阈值对应的TPR和FPR


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    """
    计算给定阈值下的分类指标

    参数:
    output: 模型输出张量
    target: 真实标签张量
    nclass: 类别数
    score_thresh: 分类阈值

    返回:
    tp: 真正例数(TP)
    pos: 实际正例数(TP + FN)
    fp: 假正例数(FP)
    neg: 实际负例数(TN + FP)
    """
    # 设置直方图参数
    mini = 1  # 类别最小值
    maxi = 1  # 类别最大值
    nbins = 1  # 直方图bin数

    # 应用sigmoid激活函数将输出转换为概率
    predict = (jt.sigmoid(output).numpy() > score_thresh).astype('int64')

    # 转换标签为int64类型
    target = target.numpy().astype('int64')

    # 计算交集: 预测正确的前景区域(TP)
    intersection = predict * (predict == target)

    # 计算真正例(TP): 预测正确的前景像素数
    tp = intersection.sum()

    # 计算假正例(FP): 预测为前景但实际是背景的像素数
    fp = (predict * (predict != target)).sum()

    # 计算真负例(TN): 预测为背景且实际是背景的像素数
    tn = ((1 - predict) * (predict == target)).sum()

    # 计算假负例(FN): 预测为背景但实际是前景的像素数
    fn = ((predict != target) * (1 - predict)).sum()

    # 计算实际正例数 = TP + FN
    pos = tp + fn

    # 计算实际负例数 = FP + TN
    neg = fp + tn

    return tp, pos, fp, neg