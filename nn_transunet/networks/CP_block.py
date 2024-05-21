# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch.nn import Module, Conv3d, Parameter, Softmax
from torch import nn


class PAM_Module(Module):
    """Location attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        Input:
        x: Input feature map (B, C, D, H, W)
        Back:
        out: Attention result + input feature
        attention: Bx(DxHxW)x(DxHxW)
        """
        m_batchsize, C, D, H, W = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, D*H*W).permute(0, 3, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, D*H*W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, D*H*W)

        out = torch.bmm(proj_value, attention.permute(0, 3, 2, 1))
        out = out.view(m_batchsize, C, D, H, W)

        out = self.gamma * out + x
        return out

class CAM_Module(Module):
    """Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        Input:
        x: Input feature map (B x C x D x H x W)
        Back:
        out£: Attention value + input feature
        attention: B x C x C
        """
        m_batchsize, C, D, H, W = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 3, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, D, H, W)

        out = self.gamma * out + x
        return out


def norm(planes, mode='bn', groups=16):
    if mode == 'bn':
        return nn.BatchNorm3d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)

        return sasc_output


class CNetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNetHead, self).__init__()
        inter_channels = in_channels // 16

        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.sc = CAM_Module(inter_channels)
        self.conv52 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())

    def forward(self, x):
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sc_conv
        sasc_output = self.conv8(feat_sum)

        return sasc_output

class PNetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PNetHead, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat_sum = sa_conv
        sasc_output = self.conv8(feat_sum)

        return sasc_output