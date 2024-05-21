import torch
import torch.nn.functional as F
import torch.nn as nn

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def downsample(x):
    return x[:, :, ::1, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2, 0, 0), mode='reflect')
    out = F.conv3d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], device=x.device)], dim=4)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] , x.shape[3]*2, x.shape[4])
    cc = cc.permute(0, 1, 2, 4, 3)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[4], x.shape[3]*2, device=x.device)], dim=4)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]*2,x.shape[4]* 2)
    x_up = cc.permute(0, 1, 2, 4, 3)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3] or up.shape[4] != img.shape[4]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3], img.shape[4]))
    diff = img - up
    return diff

def make_laplace_pyramid(img, level, channels):
    # Laplacian Pyramid
    # It is used to calculate the detail information of the image on different scales, perform Gaussian blur and downsampling on the input, recover the size of the current image in the up-sampling, and then calculate the difference between the image before processing and the image after up-sampling
    # Difference, get the details, and return a list of all the layers' Laplacian features.
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        up = up.permute(0, 1, 2, 4, 3)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3] or up.shape[4] != current.shape[4]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Out(nn.Module):
    #
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv3d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(Conv(in_channels, in_channels // 4),
            Conv(in_channels // 4, out_channels)

        )
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

    def forward(self, x1, x2):
        # x2 = self.up(x2)
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)


class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        self.fusion_conv = nn.Sequential(nn.Conv3d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(nn.Conv3d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid())


    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]
        pred = torch.sigmoid(pred)

        # reverse attention
        background_att = 1 - pred
        background_x = x * background_att

        # boudary attention
        edge_pred = make_laplace(pred, 1)
        pred_feature = x * edge_pred

        # high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='trilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        return out