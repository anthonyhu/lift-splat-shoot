from collections import OrderedDict
from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from src.layers.base import Interpolate


class ConvBlock(nn.Module):
    """2D convolution followed by
         - an optional normalisation (batch norm or instance norm)
         - an optional activation (ReLU, LeakyReLU, or tanh)
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, norm='bn', activation='relu',
                 bias=False, transpose=False):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=1)
        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    """Residual block:
       x -> Conv -> norm -> act. -> Conv -> norm -> act. -> ADD -> out
         |                                                   |
          ---------------------------------------------------
    """
    def __init__(self, in_channels, out_channels=None, norm='bn', activation='relu', bias=False, dropout=0.2):
        super().__init__()
        out_channels = out_channels or in_channels

        self.layers = nn.Sequential(OrderedDict([
            ('conv_1', ConvBlock(in_channels, in_channels, 3, stride=1, norm=norm, activation=activation, bias=bias)),
            ('conv_2', ConvBlock(in_channels, out_channels, 3, stride=1, norm=norm, activation=activation, bias=bias)),
            ('dropout', nn.Dropout2d(dropout)),
        ]))

        if out_channels != in_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.projection = None

    def forward(self, x):
        x_residual = self.layers(x)

        if self.projection:
            x = self.projection(x)
        return x + x_residual


#############
# Used by monodepth
class ConvBlock2(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

################


class Bottleneck(nn.Module):
    """
    Defines a bottleneck module with a residual connection
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, dilation=1,
                 groups=1, upsample=False, downsample=False, dropout=0.2):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        # Define the main conv operation
        if dilation > 1:
            assert not downsample, f'downsample and dilation >1 not supported, got dilation: {dilation}'
            assert not upsample, f'upsample and dilation >1 not supported, got dilation: {dilation}'
        elif upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size,
                                                 bias=False, dilation=1, stride=2, output_padding=padding_size,
                                                 padding=padding_size, groups=groups)
        elif downsample:
            bottleneck_conv = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size,
                                        bias=False, dilation=dilation, stride=2, padding=padding_size, groups=groups)
        else:
            bottleneck_conv = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, bias=False,
                                        dilation=dilation, padding=padding_size, groups=groups)

        self.layers = nn.Sequential(OrderedDict([
            # First projection with 1x1 kernel
            ('conv_down_project', nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)),
            ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
                                               nn.ReLU(inplace=True))),
            # Second conv block
            ('conv', bottleneck_conv),
            ('abn', nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
                                               nn.ReLU(inplace=True))),
            # Final projection with 1x1 kernel
            ('conv_up_project', nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)),
            ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels),
                                               nn.ReLU(inplace=True))),
            # Regulariser
            ('dropout', nn.Dropout2d(p=dropout))
        ]))

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update({
                'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1),
                'bn_skip_proj': nn.BatchNorm2d(out_channels),
            })
            self.projection = nn.Sequential(projection)

    # pylint: disable=arguments-differ
    def forward(self, *args):
        x, = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x
