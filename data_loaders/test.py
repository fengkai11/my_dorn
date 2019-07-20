# import torchvision
# from torchvision import models, transforms
#
# model = models.resnet101(pretrained=True)
# import torch
# saved_state_dict = torch.load('../resnet101-5d3b4d8f.pth')
# for i in saved_state_dict:
#     print(i)

import torch
a = torch.randn((2,3))
b = torch.randn((1,3))
c = torch.cat((a,b),dim = 0)
print(a)
print(b)
print(c)