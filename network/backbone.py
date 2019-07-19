import torch
import torch.nn as nn
from torch.nn import BatchNorm2d

affine_par = True
#size was not changed
def conv3x3(in_planes,out_planes,stride = 1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,
                     pad = 1,bias=False)
class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,inplanes,planes,stride = 1,dilation = 1,downsample = None,first_dilation=1,multi_grid = 1):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplanes = False)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,
                               padding= dilation*multi_grid,dilation = dilation*multi_grid,bias = False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size=1,bias=False)
        self.bn3 = BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplanes=False)
        self.relu_inplace = nn.ReLU(inplanes=True)
        self.downsample= downsample
        self.dilation = dilation
        self.stride = stride
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu_inplace(out)
        return out



class ResNet(nn.Module):
    def __init__(self,block,layers):
        self.inplanes = 128
        super(ResNet,self).__init__()
        self.conv1 = conv3x3(3,64,stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64,64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64,128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride = 2,padding =1)
        self.relu = nn.ReLU(inplace=True)
    def _make_layer(self,block,planes,blocks,stride=1,dilation = 1,multi_grad = 1):
        downsample = None
        if stride !=1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,
                          kernel_size=1,stride=stride,bias=False),
                BatchNorm2d(planes*block.expansion,affine=affine_par)
            )


