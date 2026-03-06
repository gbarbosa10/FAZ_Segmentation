# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:16:03 2023

@author: guiti
"""
from torch.utils.data import Dataset
import numpy as np
import cv2
#from torchvision.io import read_image
import torch
from scipy.ndimage import binary_fill_holes
import lightning as L
from PIL import Image
import torchvision
from skimage import feature
import skimage
from skimage.morphology import disk, binary_dilation
from skimage.morphology import flood, flood_fill, skeletonize
from skimage.filters import threshold_mean, rank
import torchvision.transforms as T

def clean_string(s):
    # Remove the prefix before the first '/'
    s = s.split('/')[-1]
    # Remove the suffix after the last '.'
    s = s.rsplit('.', 1)[0]
    return s

def draw_mask(image, mask_generated, color_array) :
    masked_image = image.copy()
    masked_image = np.where(mask_generated.astype(np.uint8),
                            color_array,
                            masked_image)
    
    masked_image = masked_image.astype(np.uint8)
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

def get_largest_contour(contour_list):
    cnt_len = 0
    position = 0
    for num, cnt in enumerate(contour_list):
        if cnt_len < cv2.arcLength(cnt,True):
            position = num
        cnt_len = cv2.arcLength(cnt,True)
    return contour_list[position]

def get_im_contour(image):
    transform = T.ToPILImage()
    image =  np.asarray(transform(image)).astype(np.uint8)
    threshold_global_otsu = threshold_mean(image*255)
    global_otsu = image*255 >= threshold_global_otsu    #global_otsu.astype(np.uint8)
    contours, hierarchy = cv2.findContours(global_otsu.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = get_largest_contour(contours)
    blank_im = np.zeros(image.shape) 
    cv2.drawContours(blank_im, cnt, -1, (255, 255, 255), 3)
    return torch.Tensor(blank_im).unsqueeze(0)

class Dataset_bg_mask(Dataset):
   def __init__(self, annotations_file, img_dir, mean, std, data_loader_num, transform=None):  
       self.img_labels = annotations_file
       self.transform = transform
       self.img_dir = img_dir
       self.mean = mean
       self.std = std
       self.dataset_index = ''
       
   def __len__(self):
       return len(self.img_labels)

   def __getitem__(self, idx):
       mean_ = self.mean
       std_ = self.std
       
       img_path = self.img_labels["filename"].iloc[idx]
       self.dataset_index = clean_string(img_path)
       image = Image.open(self.img_dir + img_path)
       image = torchvision.transforms.functional.pil_to_tensor(image)
       label = Image.open(self.img_dir + self.img_labels["label"].iloc[idx])
       
       label = torchvision.transforms.functional.pil_to_tensor(label)
       label = label[0:1, :, :] > 128
       label = label.float()
       
       image=image/255   
       image1 = (image - mean_) / (std_)

       if image.shape[0] == 1:
            image = torch.concatenate((image1, image1), axis=0)
            image = torch.concatenate((image, image1), axis=0)
       if self.transform:
            image = self.transform(image)
       return image, self.transform(torch.Tensor(label)), self.dataset_index

class Dataset_bg_mask_oftal_2(Dataset):
    def __init__(self, annotations_file, img_dir, mean, std, data_loader_num, transform=None):
        self.img_labels = annotations_file
        self.transform = transform
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        self.dataset_index = ''

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        mean_ = self.mean
        std_ = self.std

        img_path = self.img_labels["filename"].iloc[idx]
        self.dataset_index = clean_string(img_path)
        image = Image.open(self.img_dir + img_path)
        image = torchvision.transforms.functional.pil_to_tensor(image)

        label = Image.open(self.img_dir + self.img_labels["label"].iloc[idx])
        label = torchvision.transforms.functional.pil_to_tensor(label)
        label = label[0:1, :, :] > 128
        label = label.float()

        label_2 = Image.open(self.img_dir + self.img_labels["label_2"].iloc[idx])
        label_2 = torchvision.transforms.functional.pil_to_tensor(label_2)
        label_2 = label_2[0:1, :, :] > 128
        label_2 = label_2.float()
        
        image = image / 255
        image1 = (image - mean_) / (std_)

        if image.shape[0] == 1:
            image = torch.concatenate((image1, image1), axis=0)
            image = torch.concatenate((image, image1), axis=0)
        if self.transform:
            image = self.transform(image)
        return image, self.transform(torch.Tensor(label)), self.transform(torch.Tensor(label_2)), self.dataset_index

class Dataset_pre_train(Dataset):
   def __init__(self, annotations_file, img_dir, mean, std, data_loader_num, transform=None):  
       self.img_labels = annotations_file
       self.transform = transform
       self.img_dir = img_dir
       self.mean = mean
       self.std = std
       self.dataset_index = ''
       
   def __len__(self):
       return len(self.img_labels)

   def __getitem__(self, idx):
       mean_ = self.mean
       std_ = self.std
       
       img_path = self.img_labels["filename"].iloc[idx]
       self.dataset_index = clean_string(img_path)
       image = Image.open(self.img_dir + img_path)
       image = torchvision.transforms.functional.pil_to_tensor(image)
       label = Image.open(self.img_dir + self.img_labels["label"].iloc[idx])

       label = torchvision.transforms.functional.pil_to_tensor(label)
       label = label[0:1, :, :] > 128
       label = label.float()

       image=image/255   
       image1 = (image - mean_) / (std_)

       if image.shape[0] == 1:
            image = torch.concatenate((image1, image1), axis=0)
            image = torch.concatenate((image, image1), axis=0)
       if self.transform:
            image = self.transform(image)
       
       # Create mask with six block defect regions
       n_obs = np.random.randint(1, 10, size=1)[0]
       mask = np.zeros((image.shape[1], image.shape[2]) , dtype=bool)
       for obs in range(n_obs):
            center_x , center_y = int(image.shape[1]/2), int(image.shape[2]/2)
            center_x = center_x + np.random.randint(20, size=1)[0]*(-1)**(np.random.randint(0, 1, size=1)[0])
            center_y = center_y + np.random.randint(20, size=1)[0]*(-1)**(np.random.randint(0, 1, size=1)[0])
            circle_or_square = 0
            if circle_or_square == 0:
                len = np.random.randint(100, size=1)[0]
                wid = np.random.randint(100, size=1)[0]
                mask[center_x - int(len/2):center_x+int(len/2), center_y - int(wid/2):center_y+int(wid/2)] = 1
       
       # Apply defect mask to the image over the same region in each color channel
       final_mask = np.concatenate((np.reshape(mask, (1, mask.shape[0], mask.shape[1])), np.reshape(mask, (1, mask.shape[0], mask.shape[1])), np.reshape(mask, (1, mask.shape[0], mask.shape[1]))), axis=0)
       image_defect = image * ~final_mask

       return image_defect, image, final_mask, self.dataset_index

class Dataset_bg_mask_cont(Dataset):
   def __init__(self, annotations_file, img_dir, mean, std, data_loader_num, transform=None):  
       self.img_labels = annotations_file
       self.transform = transform
       self.img_dir = img_dir
       self.mean = mean
       self.std = std
       self.dataset_index = ''
       
   def __len__(self):
       return len(self.img_labels)

   def __getitem__(self, idx):
       mean_ = self.mean
       std_ = self.std
       
       img_path = self.img_labels["filename"].iloc[idx]
       self.dataset_index = clean_string(img_path)
       image = Image.open(self.img_dir + img_path) #[9:]) # +  self.img_dir
       image = torchvision.transforms.functional.pil_to_tensor(image)
       label = Image.open(self.img_dir + self.img_labels["label"].iloc[idx])#[9:]) # "C:/Users/gbarbosa/Documents/INEGI/Blockchain/OCT_diganosis/vm_code/segmentation/" +
       
       label = torchvision.transforms.functional.pil_to_tensor(label)
       label = label[0:1, :, :] > 128
       label = label.float()
       
       image=image/255   
       image1 = (image - mean_) / (std_)

       if image.shape[0] == 1:
            image = torch.concatenate((image1, image1), axis=0)
            image = torch.concatenate((image, image1), axis=0)
       if self.transform:
            image = self.transform(image)
       return image, self.transform(torch.Tensor(label)), self.dataset_index