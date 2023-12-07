import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from pvtv2 import pvt_v2_b2
from resnet import ResNet50, ResNet101
import cv2
import numpy as np
import matplotlib.pyplot as plt


def weight_init(module):
    """
    Initialize the weights of the given module.

    Args:
        module (nn.Module): The module to initialize the weights for.
    """
    for n, m in module.named_children():
        # print('initialize: ' + n)
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.PReLU, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
            pass
        elif isinstance(m, ops.DeformConv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.weight is not None:
                nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()



class Fusion(nn.Module):
    def __init__(self, channels, out_channels):
        """
        Initialize the Fusion module with the given channels and out_channels.

        Args:
            channels (list): A list of input channel sizes.
            out_channels (int): The number of output channels.
        """
        super(Fusion, self).__init__()
        self.linear2 = nn.Sequential(nn.Conv2d(channels[1], out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d(channels[2], out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[3], out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3, x4):
        """
        Forward pass of the Fusion module.

        Args:
            x1, x2, x3, x4 (torch.Tensor): Input tensors.

        Returns:
            torch.Tensor: The fused output tensor.
        """
        x2, x3, x4 = self.linear2(x2), self.linear3(x3), self.linear4(x4)
        x4 = F.interpolate(x4, size=x1.size()[2:], mode='bilinear')
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear')
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear')
        out = x2 + x3 + x4
        return out

    def initialize(self):
        """
        Initialize the model weights.
        """
        weight_init(self)





class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1      = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1        = nn.BatchNorm2d(out_channels)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(out_channels)
        self.shortcut   = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.initialize()

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(res)
        out = self.relu(out)
        return out 

    def initialize(self):
        """
        Initialize the model weights.
        """
        weight_init(self)

class DeformableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 iter,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv, self).__init__()
        
        assert isinstance(kernel_size, tuple) or isinstance(kernel_size, int)

        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        if iter:
            self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=self.stride,
                                     padding=self.padding, 
                                     bias=True)
        else:
            self.offset_conv = nn.Conv2d(2*in_channels, 
                                        2 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size, 
                                        stride=self.stride,
                                        padding=self.padding, 
                                        bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=self.stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      bias=bias)
        self.initialize()

    def initialize(self):
        weight_init(self)

    def forward(self, x, residual):
        offset = self.offset_conv(residual)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias, padding=self.padding, mask=modulator, stride=self.stride)
        return x


class FAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        This class implements the Feature Aggregation Module (FAM) for image super-resolution.
        It takes in two inputs, a low-resolution image and a high-resolution image, and outputs a high-resolution image.
        The FAM uses either a flow-based or deformable-based fusion method to combine the features from the two inputs.
        """
        super(FAM, self).__init__()
        self.fuse = 'flow'
        self.iter = False
        self.down_h = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.down_l = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.fuse == 'flow':
            self.flow_make = nn.Conv2d(out_channels*2, 2, kernel_size=3, padding=1, bias=False)
            if self.iter:
                self.iter_flow_make = nn.Conv2d(out_channels, 2, kernel_size=3, padding=1, bias=False)
        elif self.fuse == 'deform':
            self.deform_conv1 = DeformableConv(in_channels, out_channels, iter=False)
            self.deform_conv2 = DeformableConv(in_channels, out_channels, iter=False)
            if self.iter:
                self.deform_conv3 = DeformableConv(in_channels, out_channels, self.iter)
                self.deform_conv4 = DeformableConv(in_channels, out_channels, self.iter)
        else:
            raise ValueError("no this type of fuse method")
        self.initialize()

    def flow_warp(self, input, flow, size):
        """
        This function warps the input image using the given flow field.
        """
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, x, y):
        low_feat_ori    = x
        high_feat_ori   = y
        h, w            = y.size()[2:]
        size            = (h, w)
        low_feat        = self.down_l(x)
        high_feat       = self.down_h(y)
        low_feat        = F.interpolate(low_feat, size=size, mode='bilinear', align_corners=True)
        if self.fuse == 'flow':
            flow        = self.flow_make(torch.cat([high_feat, low_feat], 1))
            b, c, hh, ww = flow.shape

            #### draw ####
            # flow_copy = torch.mean(flow, dim=1)[0].cpu().detach().numpy()
            # plt.imshow(flow_copy,cmap='jet')
            # plt.savefig('flow')
            
            low_feat_warp   = self.flow_warp(low_feat_ori, flow, size=size)
            high_feat_warp  = self.flow_warp(high_feat_ori, flow, size=size)
            if self.iter:
                iter_flow       = self.iter_flow_make(low_feat_warp+high_feat_warp)
                low_feat_warp   = self.flow_warp(low_feat_ori, iter_flow, size=size)
                high_feat_warp  = self.flow_warp(high_feat_ori, iter_flow, size=size)
        elif self.fuse == 'deform':
            low_feat_ori        = F.interpolate(low_feat_ori, size=size, mode='bilinear', align_corners=True)
            low_feat_warp       = self.deform_conv1(low_feat_ori, torch.cat([high_feat, low_feat], 1))
            high_feat_warp      = self.deform_conv2(high_feat_ori, torch.cat([high_feat, low_feat], 1))
            if self.iter:
                inter_feat      = low_feat_warp + high_feat_warp
                low_feat_warp   = self.deform_conv3(low_feat_ori, inter_feat)
                high_feat_warp  = self.deform_conv4(high_feat_ori, inter_feat)
        
        out = high_feat_warp + low_feat_warp 
        return out


    def initialize(self):
        weight_init(self)



class FAFPN(nn.Module):
    def __init__(self, channels):
        super(FAFPN, self).__init__()
        c1, c2, c3, c4  = channels
        self.conv1      = ResBlock(c1, c1)
        self.up1        = ResBlock(c2, c1)
        self.conv2      = ResBlock(c2, c2)
        self.up2        = ResBlock(c3, c2)
        self.conv3      = ResBlock(c3, c3)
        self.up3        = ResBlock(c4, c3)

        self.fuse1      = FAM(c1, c1)
        self.fuse2      = FAM(c2, c2)
        self.fuse3      = FAM(c3, c3)
        self.initialize()

    def forward(self, x1, x2, x3, x4):
        out = self.fuse3(self.up3(x4), self.conv3(x3))
        out = self.fuse2(self.up2(out), self.conv2(x2))
        out = self.fuse1(self.up1(out), self.conv1(x1))

        return out

    def initialize(self):
        weight_init(self)

class CenterNet(nn.Module):
    def __init__(self, args):
        super(CenterNet, self).__init__()
        channels           = [64, 128, 320, 512]
        if args.backbone == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
        elif args.backbone == 'resnet50':
            self.backbone = ResNet50()
            channels      = [256, 512, 1024, 2048]

        self.args     = args
        self.fusion   = Fusion(channels, 64)
        # 热力图预测部分
        self.cls_head = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 1, kernel_size=1))
        # 宽高预测的部分
        self.wh_head  = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 2, kernel_size=1))
        # 中心点预测的部分
        self.reg_head = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 2, kernel_size=1))
        self.initialize()

        
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x) # 512,512,3 -> 16,16,2048
        pred           = self.fusion(x1, x2, x3, x4)
        heatmap        = torch.sigmoid(self.cls_head(pred))
        whpred         = self.wh_head(pred)
        offset         = self.reg_head(pred)
        return heatmap, whpred, offset
    
    def initialize(self):
        if self.args.snapshot:
            self.load_state_dict(torch.load(self.args.snapshot))
        else:
            weight_init(self)




class ECCPolypDet(nn.Module):
    def __init__(self, args):
        """
        Initialize the ECCPolypDet with the given arguments.

        Args:
            args: A namespace containing the model configuration.
        """
        super(ECCPolypDet, self).__init__()
        channels = [64, 128, 320, 512]
        backbone_options = {
            'pvt_v2_b2': (pvt_v2_b2, channels),
            'resnet50': (ResNet50, [256, 512, 1024, 2048]),
        }

        if args.backbone in backbone_options:
            print(f'backbone: {args.backbone}')
            backbone_class, channels = backbone_options[args.backbone]
            self.backbone = backbone_class()
        else:
            raise ValueError(f"Invalid backbone: {args.backbone}")

        self.args       = args
        self.fuse       = 'simple'
        out_channels    = 64
        self.fusion     = Fusion(channels, out_channels)
        # self.FAFPN      = FAFPN(channels)
        self.interconv  = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, 1, kernel_size=1))
        self.outerconv  = nn.Conv2d(1, out_channels, kernel_size=1)
        self.cls_head   = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, 1, kernel_size=1))
        self.wh_head    = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, 2, kernel_size=1))
        self.reg_head   = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, 2, kernel_size=1))
        self.initialize()
        
    def forward(self, x):
        """
        Forward pass of the ContrastivePolypModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: ct_feature, interheatmap, heatmap, whpred, offset
        """
        x1, x2, x3, x4  = self.backbone(x)
        if self.fuse == 'simple': 
            pred        = self.fusion(x1, x2, x3, x4)
        else:
            pred        = self.FAFPN(x1, x2, x3, x4)
        ct_feature      = F.interpolate(pred, x.size()[2:], mode='bilinear')
        interpred       = self.interconv(pred)
        interheatmap    = torch.sigmoid(interpred)
        outerpred       = self.outerconv(interpred)
        pred            = pred + outerpred
        heatmap         = torch.sigmoid(self.cls_head(pred))
        whpred          = self.wh_head(pred)
        offset          = self.reg_head(pred)
        return pred, ct_feature, interheatmap, heatmap, whpred, offset

    def initialize(self):
        """
        Initialize the model weights.
        """
        if self.args.pretrained:
            self.load_state_dict(torch.load(self.args.pretrained))
        else:
            weight_init(self)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone'   , type=str   , default='pvt_v2_b2'              )
    parser.add_argument('--snapshot'   , type=str   , default=None                     )
    parser.add_argument('--epoch'      , type=int   , default=20                       )
    parser.add_argument('--lr'         , type=float , default=1e-4                     )
    parser.add_argument('--batch_size' , type=int   , default=16                       )
    parser.add_argument('--data_path'  , type=str   , default='../dataset/'            )
    parser.add_argument('--model_path' , type=str   , default='./model/'               )
    args = parser.parse_args()
    model = CenterNet(args).cuda()
    inputs = torch.ones(2, 3, 128, 128).cuda()

    while True:
        heatmap, whpred, offset = model(inputs)
