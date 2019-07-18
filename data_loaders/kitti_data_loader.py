import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from data_loaders import transforms as my_transforms
to_tensor  = my_transforms.ToTensor()

def pil_loader(path,rgb = True):
    with open(path,'rb') as f:
        img = Image.open(f)
        if rgb:
            return img.convert('RGB')
        else:
            return img.convert('I')
def readPathFiles(file_path,root_dir):
    im_gt_pahts = []
    with open(file_path) as fr:
        lines = fr.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            im_path = os.path.join(root_dir,line.split()[0])
            gt_path = os.path.join(root_dir,line.split()[1])
            im_gt_pahts.append((im_path,gt_path))
    return im_gt_pahts

class KittiFolder(Dataset):
    def __init__(self,root_dir = r'D:\my_code\project\test\data\test.jpg',mode = 'train',
                 size = (200,100),loader = pil_loader):
        super(KittiFolder,self).__init__()
        self.root_dir = root_dir
        self.loader = loader
        self.mode = mode
        self.size  = size
        self.img_gt_paths = None
        if self.mode == 'train':
            self.img_gt_paths = readPathFiles('train.txt',root_dir)
        elif self.mode == 'test':
            self.img_gt_paths = readPathFiles('test.txt',root_dir)
        elif self.mode == 'val':
            self.img_gt_paths = readPathFiles('val.txt',root_dir)
        else:
            print('no mode name as {}'.format(mode))
            exit(-1)
    def train_transform(self,im,gt):
        im = np.array(im).astype(np.float32)
        gt = np.array(im).astype(np.float32)
        s = np.random.uniform(1.0,1.5)
        angle = np.random.uniform(-5.0,5.0)
        do_flip = np.random.uniform(0.0,1.0)<0.5
        # color_jitter = my_transform.ColorJitter(0.4,0.4,0.4)
        transform = my_transforms.Compose([
            my_transforms.Crop(130,10,240,1200),
            my_transforms.Resize(460/240,interpolation='bilinear'),
            my_transforms.Rotate(angle),
            my_transforms.Resize(s),
            my_transforms.CenterCrop(self.size),
            my_transforms.HorizontalFlip(do_flip)
        ])
        im_ = transform(im)
        # im_ = color_jitter(im_)
        gt_ = transform(gt)
        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ = 100.0*s
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)
        return im_,gt_
    def __getitem__(self,idx):
        im_path,gt_path = self.img_gt_paths[idx]
        im = self.loader(im_path)
        gt = self.loader(gt_path,False)
        if self.mode == 'train':
            im,gt = self.train_transform(im,gt)
        else:
            im,gt = self.val_transform(im,gt)
        return im,gt


import torch
from tqdm import tqdm
if __name__ == '__main__':
    root_dir = ''
    data_set = KittiFolder(root_dir,mode = 'train',size=(200,200))
    data_loader = torch.utils.data.DataLoader(data_set,batch_size=1,shffle=False,num_workers=0)
    print('data_set num is',len(data_loader))
    for im ,gt in tqdm(data_loader):
        print(im.size())




