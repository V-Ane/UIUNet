import jittor as jt
from jittor import nn, Module

from RSU import _upsample_like
from fusion import *
from  RSU import *

class UIUNET(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super(UIUNET,self).__init__()
        # --- 编码器路径 ---
        # Stage 1: 输入图像->64通道
        self.stage1 = RSU7(in_ch, 32, 64)  # RSU7模块
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 下采样2倍

        # Stage 2: 64->128通道
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Stage 3: 128->256通道
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Stage 4: 256->512通道
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Stage 5: 512->512通道 (使用全卷积RSU4F)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 最后的下采样

        # Stage 6: 最底层特征 (RSU4F)
        self.stage6 = RSU4F(512, 256, 512)

        # --- 解码器路径 ---
        # Stage 5d: 输入1024通道(512解码+512编码)->512通道
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)  # 1024输入(512解码+512编码)->256输出
        self.stage3d = RSU5(512, 64, 128)  # 512输入(256解码+256编码)->128输出
        self.stage2d = RSU6(256, 32, 64)  # 256输入(128解码+128编码)->64输出
        self.stage1d = RSU7(128, 16, 64)  # 128输入(64解码+64编码)->64输出

        # --- 侧输出层 (多尺度监督) ---
        # 每个输出层后接1x1卷积产生分割图
        self.side1 = nn.Conv(64, out_ch, 3, padding=1)  # 最浅层输出
        self.side2 = nn.Conv(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv(512, out_ch, 3, padding=1)  # 最深层输出

        # --- 最终输出融合层 ---
        # 将6个侧输出融合为最终结果
        # 输入通道: 6*out_ch, 输出通道: out_ch
        self.outconv = nn.Conv(6 * out_ch, out_ch, 1)  # 1x1卷积融合通道

        # --- 特征融合模块 (使用非对称双向融合) ---
        # 在解码阶段融合编码和解码特征
        self.fuse5 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse4 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse3 = self._fuse_layer(256, 256, 256, fuse_mode='AsymBi')
        self.fuse2 = self._fuse_layer(128, 128, 128, fuse_mode='AsymBi')

        self.sigmoid = nn.Sigmoid()
    # 特征融合模块选择器
    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        if fuse_mode == 'AsymBi':
            # 使用非对称双向特征融合
            return AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            raise NameError(f"不支持的融合模式: {fuse_mode}")

    def execute(self,x):
        """前向传播"""
        hx = x  # 保存原始输入

        # ----- 编码路径 -----
        # Stage 1
        hx1 = self.stage1(hx)  # 初始特征提取 (RSU7)
        hx = self.pool12(hx1)  # 下采样2倍

        # Stage 2
        hx2 = self.stage2(hx)  # RSU6
        hx = self.pool23(hx2)  # 下采样

        # Stage 3
        hx3 = self.stage3(hx)  # RSU5
        hx = self.pool34(hx3)  # 下采样

        # Stage 4
        hx4 = self.stage4(hx)  # RSU4
        hx = self.pool45(hx4)  # 下采样

        # Stage 5 (全卷积)
        hx5 = self.stage5(hx)  # RSU4F
        hx = self.pool56(hx5)  # 最后的下采样

        # Stage 6 (最底层)
        hx6 = self.stage6(hx)  # RSU4F
        hx6up = _upsample_like(hx6, hx5)  # 上采样到hx5的尺寸，准备融合

        # ----- 解码路径（带特征融合）-----
        # 融合解码特征(hx6up)和编码特征(hx5)
        fusec51, fusec52 = self.fuse5(hx6up, hx5)  # 双向特征融合
        # 拼接融合结果 (通道维度)
        hx5d = self.stage5d(jt.contrib.concat([fusec51, fusec52], 1))
        hx5dup = _upsample_like(hx5d, hx4)  # 上采样到hx4尺寸

        # Stage 4解码
        fusec41, fusec42 = self.fuse4(hx5dup, hx4)
        hx4d = self.stage4d(jt.contrib.concat([fusec41, fusec42], 1))
        hx4dup = _upsample_like(hx4d, hx3)  # 上采样

        # Stage 3解码
        fusec31, fusec32 = self.fuse3(hx4dup, hx3)
        hx3d = self.stage3d(jt.contrib.concat([fusec31, fusec32], 1))
        hx3dup = _upsample_like(hx3d, hx2)  # 上采样

        # Stage 2解码
        fusec21, fusec22 = self.fuse2(hx3dup, hx2)
        hx2d = self.stage2d(jt.contrib.concat([fusec21, fusec22], 1))
        hx2dup = _upsample_like(hx2d, hx1)  # 上采样到初始尺寸

        # Stage 1解码
        hx1d = self.stage1d(jt.contrib.concat([hx2dup, hx1], 1))  # 最终解码特征

        # ----- 侧输出（多尺度监督）-----
        # 从不同层提取输出并上采样到相同尺寸
        d1 = self.side1(hx1d)  # 最精细尺度输出

        d22 = self.side2(hx2d)  # Stage2输出
        d2 = _upsample_like(d22, d1)  # 上采样到d1尺寸

        d32 = self.side3(hx3d)  # Stage3输出
        d3 = _upsample_like(d32, d1)

        d42 = self.side4(hx4d)  # Stage4输出
        d4 = _upsample_like(d42, d1)

        d52 = self.side5(hx5d)  # Stage5输出
        d5 = _upsample_like(d52, d1)

        d62 = self.side6(hx6)  # 最深层输出
        d6 = _upsample_like(d62, d1)

        # 融合所有侧输出 (通道维度拼接)
        d0 = self.outconv(jt.contrib.concat([d1, d2, d3, d4, d5, d6], 1))


        return (self.sigmoid(d0), self.sigmoid(d1), self.sigmoid(d2),
                self.sigmoid(d3), self.sigmoid(d4), self.sigmoid(d5),
                self.sigmoid(d6))