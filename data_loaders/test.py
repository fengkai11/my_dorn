# import torchvision
# from torchvision import models, transforms
#
# model = models.resnet101(pretrained=True)
# import torch
# saved_state_dict = torch.load('../resnet101-5d3b4d8f.pth')
# for i in saved_state_dict:
#     print(i)

import torch
import numpy as np
a = torch.randn((2,3))
b = torch.randn((1,3))
c = torch.cat((a,b),dim = 0)

t1 = np.zeros((2,3,4,4))
for i in range(3):
    t1[:,i,:,:] = t1[:,i,:,:] + i*np.ones((2,4,4))
print(t1)
