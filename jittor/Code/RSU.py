import jittor as jt
from jittor import nn, Module

def _upsample_like(src, tar):
    return jt.nn.interpolate(src, size=tar.shape[2:], mode="bilinear")

class REBNCONV(Module): #基础Conv2d+Bacthnorm2d+ReLU
    def __init__(self,in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV,self).__init__()
        self.conv1 = nn.Conv(in_ch, out_ch, 3, padding=1 * dirate,dilation=1 * dirate)
        self.bn1 = nn.BatchNorm(out_ch)
        self.relu1 = nn.ReLU()
    def execute(self,x):
        #卷积->归一化->relu激活
        return self.relu1(self.bn1(self.conv1(x)))



class RSU7(Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        #初始
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        #一层
        self.rebnconvin1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #二层
        self.rebnconvin2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #三层
        self.rebnconvin3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #四层
        self.rebnconvin4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #五层
        self.rebnconvin5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #六层
        self.rebnconvin6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        #七层
        self.rebnconvin7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        # --- 解码路径 (上采样) 输入通道为mid_ch*2因为要拼接编码路径的特征---
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        # 最后一层输出到out_ch
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        #nn.ConvTranspose2d不用是害怕无法复现
    def execute(self,x):
        hx = x
        hxin = self.rebnconvin(hx)

        #一层
        hx1 = self.rebnconvin1(hxin)  # 卷积
        hx = self.pool1(hx1)  # 下采样

        #二层
        hx2 = self.rebnconvin2(hx)
        hx = self.pool2(hx2)

        #三层
        hx3 = self.rebnconvin3(hx)
        hx = self.pool3(hx3)

        #四层
        hx4 = self.rebnconvin4(hx)
        hx = self.pool4(hx4)

        #五层
        hx5 = self.rebnconvin5(hx)
        hx = self.pool5(hx5)

        #六层
        hx6 = self.rebnconvin6(hx)
        #七层
        hx7 = self.rebnconvin7(hx6)

        # --- 解码阶段 (带跳跃连接) ---
        # 第6d层: 拼接hx7(解码特征)和hx6(编码特征)
        hx6d = self.rebnconv6d(jt.contrib.concat([hx7, hx6], dim=1))  # dim=1是通道维度
        hx6dup = _upsample_like(hx6d, hx5)  # 上采样到hx5的尺寸

        # 第5d层
        hx5d = self.rebnconv5d(jt.contrib.concat([hx6dup, hx5], dim=1))
        hx5dup = _upsample_like(hx5d, hx4)  # 上采样

        # 第4d层
        hx4d = self.rebnconv4d(jt.contrib.concat([hx5dup, hx4], dim=1))
        hx4dup = _upsample_like(hx4d, hx3)

        # 第3d层
        hx3d = self.rebnconv3d(jt.contrib.concat([hx4dup, hx3], dim=1))
        hx3dup = _upsample_like(hx3d, hx2)

        # 第2d层
        hx2d = self.rebnconv2d(jt.contrib.concat([hx3dup, hx2], dim=1))
        hx2dup = _upsample_like(hx2d, hx1)

        # 第1d层
        hx1d = self.rebnconv1d(jt.contrib.concat([hx2dup, hx1], dim=1))

        # 残差连接: 原始输入 + 解码输出
        return hx1d + hxin

class RSU6(Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        #一层
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #二层
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #三层
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #四层
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def execute(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        #一层
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        #二层
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        #三层
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        #四层
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        #五层
        hx5 = self.rebnconv5(hx)
        #六层
        hx6 = self.rebnconv6(hx5)


        # 第5d层
        hx5d = self.rebnconv5d(jt.contrib.concat([hx6, hx5], dim=1))
        hx5dup = _upsample_like(hx5d, hx4)  # 上采样

        # 第4d层
        hx4d = self.rebnconv4d(jt.contrib.concat([hx5dup, hx4], dim=1))
        hx4dup = _upsample_like(hx4d, hx3)

        # 第3d层
        hx3d = self.rebnconv3d(jt.contrib.concat([hx4dup, hx3], dim=1))
        hx3dup = _upsample_like(hx3d, hx2)

        # 第2d层
        hx2d = self.rebnconv2d(jt.contrib.concat([hx3dup, hx2], dim=1))
        hx2dup = _upsample_like(hx2d, hx1)

        # 第1d层
        hx1d = self.rebnconv1d(jt.contrib.concat([hx2dup, hx1], dim=1))
        return hx1d + hxin

class RSU5(Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        #一层
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #二层
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #三层
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)


    def execute(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        #一层
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        #二层
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        #三层
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        #四层
        hx4 = self.rebnconv4(hx)

        #五层
        hx5 = self.rebnconv5(hx4)


        # 第4d层
        hx4d = self.rebnconv4d(jt.contrib.concat([hx5, hx4], dim=1))
        hx4dup = _upsample_like(hx4d, hx3)

        # 第3d层
        hx3d = self.rebnconv3d(jt.contrib.concat([hx4dup, hx3], dim=1))
        hx3dup = _upsample_like(hx3d, hx2)

        # 第2d层
        hx2d = self.rebnconv2d(jt.contrib.concat([hx3dup, hx2], dim=1))
        hx2dup = _upsample_like(hx2d, hx1)

        # 第1d层
        hx1d = self.rebnconv1d(jt.contrib.concat([hx2dup, hx1], dim=1))
        return hx1d + hxin

class RSU4(Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        #一层
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #二层
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #三层
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)


        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)


    def execute(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        #一层
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        #二层
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        #三层
        hx3 = self.rebnconv3(hx)

        #四层
        hx4 = self.rebnconv4(hx3)




        # 第3d层
        hx3d = self.rebnconv3d(jt.contrib.concat([hx4, hx3], dim=1))
        hx3dup = _upsample_like(hx3d, hx2)

        # 第2d层
        hx2d = self.rebnconv2d(jt.contrib.concat([hx3dup, hx2], dim=1))
        hx2dup = _upsample_like(hx2d, hx1)

        # 第1d层
        hx1d = self.rebnconv1d(jt.contrib.concat([hx2dup, hx1], dim=1))
        return hx1d + hxin

class RSU4F(nn.Module): #无池化 全卷积
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        # 全卷积版本，使用不同dilation rate代替池化
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        # 所有层都不下采样
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)  # 扩大感受野
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)  # 最大空洞率

        # 解码路径
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)  # 初始卷积

        # 编码（无下采样）
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)  # 最底层

        # 解码（跳跃连接）
        # 从底层开始向上解码
        hx3d = self.rebnconv3d(jt.contrib.concat([hx4, hx3], dim=1))
        hx2d = self.rebnconv2d(jt.contrib.concat([hx3d, hx2], dim=1))
        hx1d = self.rebnconv1d(jt.contrib.concat([hx2d, hx1], dim=1))

        return hx1d + hxin