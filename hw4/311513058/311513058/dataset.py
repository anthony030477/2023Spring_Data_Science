import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from PIL import Image
import os
from torchvision.utils import save_image
import math
from torchvision.transforms import functional as F
import random
import numpy as np
# crop_img_size=512
# gaussion_kernel=51
transform = transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
tran_together = transforms.Compose([
            # transforms.RandomCrop((512,512),pad_if_needed=True),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation(30),
            #transforms.RandomVerticalFlip(p=0.5)
        ])

class trainDataset(Dataset):
    def __init__(self, image_dir='data/data_processed/train', points_dir='data/data_processed/train',transform=transform,train=True):
        self.image_dir = image_dir
        self.points_dir = points_dir
        if train ==True:
            self.image_filenames = sorted([file for file in os.listdir(self.image_dir) if file.endswith('.jpg')])[:3700]
            self.points_filenames = sorted([file for file in os.listdir(self.points_dir) if file.endswith('.npy')])[:3700]
        else:
            self.image_filenames = sorted([file for file in os.listdir(self.image_dir) if file.endswith('.jpg')])[3700:]
            self.points_filenames = sorted([file for file in os.listdir(self.points_dir) if file.endswith('.npy')])[3700:]
        self.transform=transform
        self.train=train
    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self, index):
        
        image_path = os.path.join(self.image_dir, self.image_filenames[index])
        image = Image.open(image_path).convert("RGB")#read_image(image_path, mode=ImageReadMode.RGB)/255.#Image.read(image_path).convert("RGB")
        
        origi_w, origi_h = image.size
        
        image = self.transform(image)
        points_path = os.path.join(self.points_dir, self.points_filenames[index])
        points = np.load(points_path)
        
        # label_path = os.path.join(self.label_dir, self.label_filenames[index])
        
        # with open(label_path, 'r') as f:
        #     location=f.readlines()#list
        #     total_label = len(location)
        # # print(location)
        # x_location=[]
        # y_location=[]
       
        # for x_y in  location:
        #     x=round(float(x_y.split(' ')[0]))#/origi_w*256
        #     x_location.append(x)
        #     y=round(float(x_y.split(' ')[1]))#/origi_h*256
        #     y_location.append(y)
        
        
        # heat_map= torch.zeros((1,origi_h,origi_w))#3x512x512
        # for i in range(len(x_location)):
        #     if y_location[i]<origi_h and x_location[i] <origi_w:
        #         heat_map[:, y_location[i], x_location[i]] = 1
            
        # blur = transforms.GaussianBlur(gaussion_kernel)
        # heat_map = blur(heat_map)
        if self.train==True:
            x_tilda=random.randint(0,origi_w-512)
            y_tilda=random.randint(0,origi_h-512)
            image=F.crop(image,y_tilda,x_tilda,512,512)
            if len(points)>0:
                idx_mask = (points[:, 1] >= y_tilda) *\
                    (points[:, 1] <= y_tilda+512) * (points[:, 0] >= x_tilda) * (points[:, 0] <= x_tilda+512)
                points = points[idx_mask]
                points[:,1]-=y_tilda
                points[:,0]-=x_tilda
            points=torch.from_numpy(points).float()
            label=torch.tensor(points.shape[0])
            # cat=torch.cat([image,heat_map],dim=0)
            # cat=tran_together(cat)
            # image=cat[0:3,:,:]
            # heat_map=cat[-1,:,:][None,:,:]

            # label=torch.sum(heat_map)
        else:
            x_tilda=random.randint(0,origi_w-512)
            y_tilda=random.randint(0,origi_h-512)
            image=F.crop(image,y_tilda,x_tilda,512,512)
            if len(points)>0:
                idx_mask = (points[:, 1] >= y_tilda) *\
                    (points[:, 1] <= y_tilda+512) * (points[:, 0] >= x_tilda) * (points[:, 0] <= x_tilda+512)
                points = points[idx_mask]
                points[:,1]-=y_tilda
                points[:,0]-=x_tilda
            points=torch.from_numpy(points).float()
            label=torch.tensor(points.shape[0])
            # w,h=int(math.ceil(origi_w/512)*512), (math.ceil(origi_h/512)*512)
            # # print(w,h)
            # target_shape = (h, w)
            # width_padding = target_shape[1] - origi_w
            # height_padding = target_shape[0] - origi_h
            # left = width_padding // 2
            # right = width_padding - left
            # top = height_padding // 2
            # bottom = height_padding - top
            # image= F.pad(image, (left, top, right, bottom))
            
            # patches = image.unfold(1, 512, 512).unfold(2, 512, 512)
            # image = patches.contiguous().view(-1, 3,512, 512)
            # points=torch.from_numpy(points).float()
            # label=torch.tensor(points.shape[0])
            # heat_map=F.pad(heat_map, (left, top, right, bottom))
            # patches=heat_map.unfold(1, 512, 512).unfold(2, 512, 512)
            # heat_map=patches.contiguous().view(-1, 1, 512, 512)
            # label=total_label
            return image,label
        st_size=min(origi_w, origi_h)
        return image,points,st_size,label
    
class testDataset(Dataset):
    def __init__(self, image_dir='test',transform=transform):
        self.image_dir = image_dir
       
        self.image_filenames = sorted([file for file in os.listdir(self.image_dir) if file.endswith('.jpg')])
        
        self.transform=transform
        
    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self, index):
        
        image_path = os.path.join(self.image_dir, self.image_filenames[index])
        image = Image.open(image_path).convert("RGB")
        
        image = self.transform(image)
        # patches = image.unfold(1, 256, 256).unfold(2, 256, 256)
        # patches = patches.contiguous().view(-1, 3, 256, 256)
        return image, 0
    


if __name__=='__main__':
    dataset=trainDataset(train=False)
    img=dataset[3][0]
    print(img.shape)
    #save_image(img, 'img1.png')
    # print(torch.sum(img))