
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import math
import torch.nn.functional as F
from torch import nn



class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        # self.dim = [512, 256, 256]
        self.dim = [256, 256, 256]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            # self.expand = add_conv(self.inter_dim, 1024, 3, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        elif level == 1:
            # self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.compress_level_0 = add_conv(256, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            # self.expand = add_conv(self.inter_dim, 512, 3, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        elif level == 2:
            # self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.compress_level_0 = add_conv(256, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(
            compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        # import ipdb
        # ipdb.set_trace()
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized = F.interpolate(
                x_level_1, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] +\
            level_1_resized * levels_weight[:, 1:2, :, :] +\
            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage



class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, top_blocks=None, use_asff = False
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        self.asff = use_asff
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks
        if self.asff:
            self.asff_level0 = ASFF(level=0)
            self.asff_level1 = ASFF(level=1)
            self.asff_level2 = ASFF(level=2)

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            # inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = F.interpolate(
                last_inner, size=(
                    int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])),
                mode='nearest'
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if self.asff:
            results_after_asff = [None, None, None]
            results_after_asff[2] = self.asff_level0(
                results[2], results[1], results[0])
            results_after_asff[1] = self.asff_level1(
                results[2], results[1], results[0])
            results_after_asff[0] = self.asff_level2(
                results[2], results[1], results[0])

            if isinstance(self.top_blocks, LastLevelP6P7):
                last_results = self.top_blocks(x[-1], results[-1])
                # results.extend(last_results)
                results_after_asff.extend(last_results)
            elif isinstance(self.top_blocks, LastLevelMaxPool):
                last_results = self.top_blocks(results[-1])
                # results.extend(last_results)
                results_after_asff.extend(last_results)
            return tuple(results_after_asff)

        else:
            if isinstance(self.top_blocks, LastLevelP6P7):
                last_results = self.top_blocks(x[-1], results[-1])
                results.extend(last_results)
            elif isinstance(self.top_blocks, LastLevelMaxPool):
                last_results = self.top_blocks(results[-1])
                results.extend(last_results)

            return tuple(results)

class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
    

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())

        gate_channels = [gate_channel]  # eg 64
        gate_channels += [gate_channel // reduction_ratio] * num_layers  # eg 4
        gate_channels += [gate_channel]  # 64
        # gate_channels: [64, 4, 4]

        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                'gate_c_fc_%d' % i,
                nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1),
                                   nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())

        self.gate_c.add_module('gate_c_fc_final',
                               nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialGate(nn.Module):
    def __init__(self,
                 gate_channel,
                 reduction_ratio=16):
        super(SpatialGate, self).__init__()
        self.branch1x1 = BasicConv2d(gate_channel, gate_channel//reduction_ratio // 4, kernel_size=3, padding = 1)

        self.branch3x1 = BasicConv2d(gate_channel, gate_channel//reduction_ratio // 4, kernel_size=3, dilation=(3,1), padding=(3,1))

        self.branch1x3 = BasicConv2d(gate_channel, gate_channel//reduction_ratio // 4, kernel_size=3, dilation=(1,3), padding=(1,3))

        self.branch3x3 = BasicConv2d(gate_channel, gate_channel//reduction_ratio // 4, kernel_size=3, dilation=3, padding=3)
    
        self.conv = nn.Sequential(
            BasicConv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, padding=1),
            nn.BatchNorm2d(gate_channel // reduction_ratio),
            nn.ReLU(),
            BasicConv2d(gate_channel//reduction_ratio, 1, kernel_size=1),
        )


    def forward(self, x):
        outs = [self.branch1x1(x),self.branch3x1(x),self.branch1x3(x),self.branch3x3(x)]
        out = torch.cat(outs, 1)
        return self.conv(out).expand_as(x)


class Feature_Enhance(nn.Module):
    def __init__(self, mode, in_dim=256):
        super(Feature_Enhance, self).__init__()
        self.mode = mode    # cls/reg
        self.channel_att = ChannelGate(gate_channel=in_dim)
        self.spatial_att = SpatialGate(gate_channel=in_dim)

    def forward(self, x, process=None):
        res = []
        for level,fp in enumerate(x) :
            atten_ft = self.channel_att(fp) * self.spatial_att(fp)
            atten = torch.sigmoid(atten_ft)
            at_mean = atten.mean()
            thres = self.calc_polar_param(process,at_mean) if process is not None else 1.0
            assert self.mode in ['cls', 'reg'], 'Choose right mode for  FE !(cls or reg)'
            if self.mode == 'reg':
                atten = torch.where(atten>thres, 1-atten, atten)
            elif self.mode == 'cls':
                if thres < 1.0:
                    atten = torch.ones_like(atten)/(torch.ones_like(atten) + torch.exp(-15*(atten_ft-0.5)))
            out = atten*fp + fp + atten_ft 
            res.append(out)
            
        return tuple(res)

    def calc_polar_param(self, process, mean):
        if process < 0.5 and process > 0.3:
            thres = mean
        else:
            thres = 1.0
        return thres


class BasicConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)

        return x
