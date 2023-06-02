import os
import numpy as np
from skimage.transform import resize
from scipy.ndimage import rotate, zoom
import random


def norm_01(x):
    return np.nan_to_num((x - np.amin(x, axis=(1, 2, 3), keepdims=True)) / (
            np.amax(x, axis=(1, 2, 3), keepdims=True) - np.amin(x, axis=(1, 2, 3), keepdims=True)))


class dataGenerator_allStack:
    def __init__(self, data_dir, data_list, batch_size, augment):
        
        # input the loaction of the data
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        self.aug = augment
        
    def _rescale(self, imageStack, MIN=0, MAX=1):
        
        if imageStack[0].max() !=1:
            
            ImageScale = []

            for stack in range(imageStack.shape[0]):
                temp = imageStack[stack,...]
                tempScale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
                ImageScale.append(tempScale.astype('float64'))
        else:
            ImageScale = imageStack
        return np.asarray(ImageScale)
    
    def _aug_img(self, img, msk):
        
        # Flip the image horizontally or vertically
        img_hori, msk_hori = np.fliplr(img), np.fliplr(msk)
        img_ver, msk_ver = np.flipud(img), np.flipud(msk)

        # Rotate the image by a random multiple of 90 degrees
        angle = np.random.choice([90, 180, 270])
        img_rot = rotate(img, angle, axes=(2, 1),reshape=True, mode='reflect')
        msk_rot = rotate(msk, angle, axes=(2, 1),reshape=True, mode='reflect')
        
        # Choose a random transformation
        img_trans = [img_hori, img_ver, img_rot]
        msk_trans = [msk_hori, msk_ver, msk_rot]
        
        i = random.randint(0, len(img_trans)-1)
        
        return img_trans[i], msk_trans[i].astype('int')
    
        
    def imageLoader(self):

        while True:
            
            for index, dataset_name in enumerate(self.data_list):
                print('load dataset:', dataset_name) 
                temp_dataset = np.load(self.data_dir + dataset_name)
                
                imgs, msks = temp_dataset['img'],temp_dataset['mask']

                L = imgs.shape[0]  
            
                batch_start = 0
                batch_end = self.batch_size

                while batch_start < L:

                    limit = min(batch_end, L) 
                    
                    # take data out and rescale
                    
                    img_temp, msk_temp = (imgs[batch_start:limit]), (msks[batch_start:limit])
                    
                    if self.aug == True:
                        img_temp, msk_temp = self._aug_img(img_temp, msk_temp)

                    # expand dimension for the model
                    img_temp = np.expand_dims(img_temp, axis=3)
                    msk_temp = np.expand_dims(msk_temp, axis=3)

                    yield(img_temp, msk_temp) 

                    batch_start += self.batch_size 
                    batch_end += self.batch_size


class dataGenerator_allStack_ft:
    def __init__(self, data_dir, data_list, batch_size, augment):
        
        # input the loaction of the data
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        self.aug = augment
        
    def _rescale(self, imageStack, MIN=0, MAX=1):
        
        if imageStack[0].max() !=1:
            
            ImageScale = []

            for stack in range(imageStack.shape[0]):
                temp = imageStack[stack,...]
                tempScale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
                ImageScale.append(tempScale.astype('float64'))
        else:
            ImageScale = imageStack
        return np.asarray(ImageScale)
    
    def _aug_img(self, img, msk, m_msk):
        
        # Flip the image horizontally or vertically
        img_hori, msk_hori, m_msk_hori = np.fliplr(img), np.fliplr(msk), np.fliplr(m_msk)
        img_ver, msk_ver, m_msk_ver = np.flipud(img), np.flipud(msk), np.flipud(m_msk)

        # Rotate the image by a random multiple of 90 degrees
        angle = np.random.choice([90, 180, 270])
        img_rot = rotate(img, angle, axes=(2, 1),reshape=True, mode='reflect')
        msk_rot = rotate(msk, angle, axes=(2, 1),reshape=True, mode='reflect')
        m_msk_rot = rotate(m_msk, angle, axes=(2, 1),reshape=True, mode='reflect')
        
        # Choose a random transformation
        img_trans = [img_hori, img_ver, img_rot]
        msk_trans = [msk_hori, msk_ver, msk_rot]
        m_msk_trans = [m_msk_hori, m_msk_ver, m_msk_rot]
        
        i = random.randint(0, len(img_trans)-1)
        
        return img_trans[i], msk_trans[i].astype('int'), m_msk_trans[i].astype('int')
    
        
    def imageLoader(self):

        while True:
            
            for index, dataset_name in enumerate(self.data_list):
                print('load dataset:', dataset_name) 
                temp_dataset = np.load(self.data_dir + dataset_name)
                
                imgs, msks, m_msks = temp_dataset['img'],temp_dataset['mask'], temp_dataset['m_mask']

                L = imgs.shape[0]  
            
                batch_start = 0
                batch_end = self.batch_size

                while batch_start < L:

                    limit = min(batch_end, L) 
                    
                    # take data out and rescale
                    
                    img_temp, msk_temp, m_msk_temp = (imgs[batch_start:limit]), (msks[batch_start:limit]), (m_msks[batch_start:limit])
                    
                    if self.aug == True:
                        img_temp, msk_temp, m_msk_temp = self._aug_img(img_temp, msk_temp, m_msk_temp)
                    
                    # expand dimension for the model
                    img_temp = np.expand_dims(img_temp, axis=3)
                    msk_temp = np.expand_dims(msk_temp, axis=3)
                    m_msk_temp = np.expand_dims(m_msk_temp, axis=3)

                    yield(img_temp, msk_temp, m_msk_temp) # output the unpacked dataset

                    batch_start += self.batch_size 
                    batch_end += self.batch_size