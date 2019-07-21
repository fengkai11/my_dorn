import torch
import torch.nn as nn
#TODO MASKED LOSS
class ordLoss(nn.Module):
    def __init__(self):
        super(ordLoss,self).__init__()
        self.loss = 0.0
    def forward(self,ord_labels,target):
        N,C,H,W = ord_labels.size()
        ord_num =C
        self.loss = 0.0
        if torch.cuda.is_available():
            K = torch.zeros((N,C,H,W),dtype = torch.int).cuda()
            for i in range(ord_num):
                K[:,i,:,:] = K[:,i,:,:]+i*torch.ones((N,H,W),dtye = torch.int).cuda()
        else:
            K = torch.zeros((N,C,H,W),dtye = torch.int)
            for i in range(ord_num):
                K[:,i,:,:] = K[:,i,:,:]+i*torch.ones((N,H,W),dtype = torch.int)
        mask_0 = (K <= target).detach()
        mask_1 = (K > target).detach()
        one = torch.ones(ord_labels[mask_1].size())
        if torch.cuda .is_available():
            one = one.cuda()
        self.loss = self.loss + torch.sum(torch.log(torch.clamp(ord_labels[mask_0],min = 1e-8,max = 1e8))) \
                    + torch.sum(torch.log(torch.clamp(one-ord_labels[mask_1],min = 1e-8,max = 1e8)))
        N = N*H*W
        self.loss = self.loss/(-N)
        return self.loss
