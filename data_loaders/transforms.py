import scipy.ndimage.interpolation as itpl
import scipy.misc as misc
import numpy as np


class Rotate(object):
    def __init__(self,angle):
        self.angle = angle
    def __call__(self,img):
        return itpl.rotate(img,self.angle,reshape=False,prefilter=False,order = 0)

class Resize(object):
    def __init__(self,size,interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        if img.ndim == 3:
            return misc.imresize(img,self.size,self.interpolation)
        elif img.ndim == 2:
            return misc.imresize(img,self.size,self.interpolation,'F')
class Compose(object):
    def __init__(self,transforms):
        self.transforms=transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return(img)
class CenterCrop(object):
    def __init__(self,size):
        self.size = size
    @staticmethod
    def get_params(img,output_size):
        h = img.shape[0]
        w = img.shape[1]
        th,tw = output_size
        i = int(round((h-th)/2.))
        j = int(round((w-tw)/2.))
        return i,j,th,tw
    def __call__(self, img):
        i,j,h,w = self.get_params(img,self.size)
        if img.ndim == 3:
            return img[i:i+h,j:j+w,:]
        elif img.ndim == 2:
            return img[i:i+h,j:j+w]
class HorizontalFlip(object):
    def __init__(self,do_flip):
        self.do_flip = do_flip
    def __call__(self,img):
        if self.do_flip:
            return np.fliplr(img)
        else:
            return
class Crop(object):
    def __init__(self,i,j,h,w):
        self.i = i
        self.j = j
        self.h = h
        self.w = w
    def __call__(self, img):
        i,j,h,w = self.i,self.j,self.h,self.w
        if img.ndim == 3:
            return img[i:i+h,j:j+w,:]
        elif img.ndim == 2:
            return img[i:i+h,j:j+w]
    def __repr__(self):
        return self.__class__.__name__+'(i = {0},j={1},h={2},w={3})'.format(
            self.i,self.j,self.h,self.w
        )
# class ColorJitter(object):
#     def __init__(self,brightness=0, contrast=0, saturation=0, hue=0):
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation =


face = misc.face()
rotate = Rotate(45)
resize_im = Resize((500,200))
center_im = CenterCrop((150,150))
crop_im = Crop(50,50,100,100)
print(crop_im)

transforms = Compose([rotate,resize_im,center_im,crop_im])

rotate_face = transforms(face)
import  matplotlib.pyplot as plt
plt.imshow(rotate_face)
plt.axis('off')
plt.show()