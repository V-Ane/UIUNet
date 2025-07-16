import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np

class SpatialAttention(Module): #空间注意力模块
    def __init__(self,kernel_size=3):
        super(SpatialAttention,self).__init__()

        #确保卷积核大小并确定padding
        assert  kernel_size in (3,7)
        padding = 3 if kernel_size == 7 else 1

        #平均值和最大值融合为1通道
        self.conv1 = nn.Conv(2,1,kernel_size,padding = padding,bias=False)

    def execute(self,x):
        #均值和最值
        avg_out = jt.mean(x,dim=1,keepdims=True)
        max_out = jt.max(x,dim=1,keepdims=True)

        #拼接
        x = jt.contrib.concat([avg_out, max_out], dim=1)
        #融合
        x = self.conv1(x)
        return x

class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()  # 调用父类构造函数
        assert in_low_channels == out_channels

        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        #高层特征处理
        self.feature_high = nn.Sequential(
            nn.Conv(self.high_channels,self.out_channels,1,1,0),
            nn.BatchNorm(out_channels),
            nn.ReLU(),
        )

        #自上而下（通道注意力）
        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv(self.out_channels,self.bottleneck_channels,1,1,0), #压缩
            nn.BatchNorm(self.bottleneck_channels),
            nn.ReLU(),
            nn.Conv(self.bottleneck_channels, self.out_channels,1,1,0), #拓展
            nn.BatchNorm(self.out_channels),
            nn.Sigmoid(),
        )

        #自下而上
        self.bottomup = nn.Sequential(
            nn.Conv(self.low_channels,self.bottleneck_channels,1,1,0),
            nn.BatchNorm(self.bottleneck_channels),
            nn.ReLU(),
            SpatialAttention(kernel_size=3),
            nn.Sigmoid(),
        )

        #后处理
        self.post = nn.Sequential(
            nn.Conv(self.out_channels,self.out_channels,3,1,1),
            nn.BatchNorm(self.out_channels),
            nn.ReLU(),
        )

    def execute(self,xh, xl):
        xh = self.feature_high(xh)      #高层特征压缩
        topdown_wei = self.topdown(xh)  #自上而下注意力
        bottomup_wei = self.bottomup(xl * topdown_wei)

        xs1 = 2 * xl * topdown_wei
        out1 = self.post(xs1)

        xs2 = 2 * xh * bottomup_wei
        out2 = self.post(xs2)

        return out1, out2