"""
Combined Model Architecture File for Eye API
Includes:
1. CKAN-SE (ResNet18 + KAN-SE attention + KANLinear classifier) for Glaucoma
2. CNN_Retino (4-layer CNN + Pooling + Dropout) for Diabetic Retinopathy
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# =====================================================================
# 1. KAN & CKAN-SE ARCHITECTURE COMPONENTS (GLAUCOMA)
# =====================================================================

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=None):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1).contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight   = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise   = scale_noise
        self.scale_base    = scale_base
        self.scale_spline  = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps        = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler,
                                          a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid  = self.grid
        x     = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x, y):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A        = self.b_splines(x).transpose(0, 1)
        B        = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output   = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake        = self.spline_weight.abs().mean(-1)
        reg_activation = l1_fake.sum()
        p              = l1_fake / reg_activation
        reg_entropy    = -torch.sum(p * p.log())
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy


class StandardSE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        bottleneck   = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc_down = nn.Linear(channels,   bottleneck)
        self.fc_up   = nn.Linear(bottleneck, channels)

    def forward(self, x):
        s = self.squeeze(x).view(x.size(0), -1)
        s = F.relu(self.fc_down(s))
        s = torch.sigmoid(self.fc_up(s))
        return x * s.view(x.size(0), x.size(1), 1, 1)


class KAN_SE_Block(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        bottleneck    = max(1, channels // reduction)
        self.squeeze  = nn.AdaptiveAvgPool2d(1)
        self.kan_down = KANLinear(channels,   bottleneck)
        self.kan_up   = KANLinear(bottleneck, channels)

    def forward(self, x):
        s = self.squeeze(x).view(x.size(0), -1)
        s = self.kan_down(s)
        s = self.kan_up(s)
        s = torch.sigmoid(s)
        return x * s.view(x.size(0), x.size(1), 1, 1)


def get_resnet18_backbone():
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return (
        nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
        backbone.layer1, backbone.layer2, backbone.layer3,
        backbone.layer4, backbone.avgpool,
    )


class VariantD_CKAN_SE(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        stem, l1, l2, l3, l4, pool = get_resnet18_backbone()
        self.stem = stem
        self.layer1 = l1;  self.kanse1 = KAN_SE_Block(64)
        self.layer2 = l2;  self.kanse2 = KAN_SE_Block(128)
        self.layer3 = l3;  self.kanse3 = KAN_SE_Block(256)
        self.layer4 = l4;  self.kanse4 = KAN_SE_Block(512)
        self.avgpool   = pool
        self.kan_head1 = KANLinear(512, 256)
        self.dropout   = nn.Dropout(0.5)
        self.kan_head2 = KANLinear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.kanse1(self.layer1(x)); x = self.kanse2(self.layer2(x))
        x = self.kanse3(self.layer3(x)); x = self.kanse4(self.layer4(x))
        x = self.avgpool(x); x = torch.flatten(x, 1)
        x = self.kan_head1(x); x = self.dropout(x)
        return self.kan_head2(x)


# =====================================================================
# 2. CNN_RETINO ARCHITECTURE COMPONENTS (DIABETIC RETINOPATHY)
# =====================================================================

def findConv2dOutShape(hin: int, win: int, conv: nn.Conv2d, pool: int = 2):
    kernel_size = conv.kernel_size
    stride      = conv.stride
    padding     = conv.padding
    dilation    = conv.dilation

    hout = math.floor(
        (hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    wout = math.floor(
        (win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    if pool:
        hout = hout // pool
        wout = wout // pool

    return int(hout), int(wout)


class CNN_Retino(nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        Cin, Hin, Win      = params["shape_in"]
        init_f             = params["initial_filters"]
        num_fc1            = params["num_fc1"]
        num_classes        = params["num_classes"]
        self.dropout_rate  = params["dropout_rate"]

        self.conv1 = nn.Conv2d(Cin,          init_f,      kernel_size=3)
        h, w       = findConv2dOutShape(Hin, Win, self.conv1)
        self.conv2 = nn.Conv2d(init_f,       2 * init_f,  kernel_size=3)
        h, w       = findConv2dOutShape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * init_f,   4 * init_f,  kernel_size=3)
        h, w       = findConv2dOutShape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * init_f,   8 * init_f,  kernel_size=3)
        h, w       = findConv2dOutShape(h, w, self.conv4)

        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = F.relu(self.conv1(X));        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X));        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X));        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X));        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate, training=self.training)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)
