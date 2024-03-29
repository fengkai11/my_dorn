import torch
import torch.nn as nn
from network.backbone import resnet101
import math

def init_conv(m,type):
    if type == 'xavier':
        torch.nn.init.xavier_normal_(m.weight)
    elif type == 'kaiming':
        torch.nn.init.kaiming_normal_(m.weight)
    else:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    if m.bias is not None:
        m.bias.data.zero_()
def init_bn(m):
    m.weight.data.fill_(1.0)
    m.bias.data.zero_()
def init_linear(m,type):
    if type == 'xavier':
        torch.nn.init.xavier_normal_(m.weight)
    elif type == 'kaiming':
        torch.nn.init.kaiming_normal_(m.weight)
    else:
        m.weight.data.fill_(1.0)
    if m.bias is not None:
        m.bias.data.zero_()

#TODO will the function change the vale of module?
def weights_init(modules,type = 'xavier'):
    m = modules
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        init_conv(m,type)
    elif isinstance(m,nn.BatchNorm2d):
        init_bn(m)
    elif isinstance(m,nn.Linear):
        init_linear(m,type)
    elif isinstance(m,nn.Module):
        for item in modules:
            if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
                init_conv(item, type)
            elif isinstance(item, nn.BatchNorm2d):
                init_bn(item)
            elif isinstance(item, nn.Linear):
                init_linear(item, type)

class FullImageEncoder(nn.Module):
    def __init__(self):
        super(FullImageEncoder,self).__init__()
        self.global_pooling  = nn.AvgPool2d(16,stride = 16,padding = (8,8))
        self.dropout = nn.Dropout2d(p = 0.5)
        self.global_fc = nn.Linear(2048*4*5,512)#based on input
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512,512,1)
        self.upsample = nn.UpsamplingBilinear2d(size = (49,65))#KITTI 49x65
        weights_init(self.modules(),'xavier')
    def forward(self,x):
        x1 = self.global_pooling(x)
        x2 = self.dropout(x1)
        x3 = x2.view(-1,2048*4*5)
        x4 = self.relu(self.global_fc(x3))
        x4 = x4.view(-1,512,1,1)
        x5 = self.conv1(x4)
        out = self.upsample(x5)
        return out
class SceneUnderstandingModule(nn.Module):
    def __init__(self):
        super(SceneUnderstandingModule,self).__init__()
        self.encoder = FullImageEncoder()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048,512,1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048,512,3,padding = 6,dilation= 6),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048,512,3,padding = 12,dilation= 12),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048,512,3,padding = 12,dilation=12),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p = 0.5),
            nn.Conv2d(512*5,2048,1),
            nn.ReLU(inplace = True),
            nn.Dropout2d(p = 0.5),
            nn.Conv2d(2048,142,1),#kitti output
            nn.UpsamplingBilinear2d(size = (385,513))
        )
        weights_init(self.modules(),'xavier')
    def forward(self,x):
        x1 = self.encoder(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1,x2,x3,x4,x5),dim=1)
        out = self.concat_process(x6)
        return out
class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer,self).__init__()
    def forward(self,x):
        N,C,H,W = x.size()
        ord_num = C//2

        A = x[:,::2,:,:].clone()
        B = x[:,1::2,:,:].clone()

        A = A.view(N,1,ord_num*H*W)
        B = B.view(N,1,ord_num*H*W)

        C = torch.cat((A,B),dim = 1)
        C = torch.clamp(C,min= 1e-8,max = 1e8)

        ord_c = nn.functional.softmax(C,dim=1)
        ord_c1 = ord_c[:,1,:].clone()
        ord_c1 = ord_c1.view(-1,ord_num,H,W)
        decode_c = torch.sum((ord_c1>0.5),dim=1).view(-1,1,H,W)
        return decode_c,ord_c1






class DORN(nn.Module):
    def __init__(self,output_size=(257,353),channel=3,pretrained = True,freeze = True):
        super(DORN,self).__init__()

        self.output_size = output_size
        self.channel = channel
        self.feature_extractor = resnet101(pretrained = pretrained)
        self.aspp_module = SceneUnderstandingModule( )
        self.orl = OrdinalRegressionLayer()
    def forward(self,x):
        x1 = self.feature_extractor(x)
        print(x1.size())
        x2 = self.aspp_module(x1)
        print(x2.size())
        depth_labels,ord_labels = self.orl(x2)
        return depth_labels,ord_labels
    def get_1x_lr_params(self):
        b = [self.feature_extractor]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k
    def get_10x_lr_params(self):
        b = [self.aspp_module,self.orl]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

if __name__ == "__main__":
    model = DORN(pretrained=False)
    model = model.cuda()
    model.eval()
    image = torch.randn(1, 3, 385, 513)
    image = image.cuda()
    with torch.no_grad():
        out0, out1 = model(image)
    print('out0 size:', out0.size())
    print('out1 size:', out1.size())