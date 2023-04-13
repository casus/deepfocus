import os
import numpy as np
from skimage.transform import resize


def norm_01(x):
    return np.nan_to_num((x - np.amin(x, axis=(1, 2, 3), keepdims=True)) / (
            np.amax(x, axis=(1, 2, 3), keepdims=True) - np.amin(x, axis=(1, 2, 3), keepdims=True)))


# for transfer learning: input: img[t, W, H, CH=3]
class dataGenerator_transfer:
    def __init__(self, img_dir, img_list, mask_dir, mask_list, batch_size):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list
        self.batch_size = batch_size
        
    def _load_img(self,img_dir, img_list):

        images = []

        for i, image_name in enumerate(img_list):
            # could add pre-process here 
            if (image_name.split('.')[1] == 'npy'):

                image = np.load(img_dir + image_name)  # load npy if data type is correct
                # print(image.min(), image.max())

                if image.max() != 1:
                    image = image/255  # rescale into [0, 1]

                images.append(image)
            else:
                print('illegal data format')

        images = np.array(images)  # convert into array

        return images.astype('float64')  # for NN
        
    def imageLoader(self):

        L = len(self.img_list)

        # keras require generator to be infinite, so we use while true
        while True:

            batch_start = 0
            batch_end = self.batch_size

            while batch_start < L:

                limit = min(batch_end, L) 
                
                X = self._load_img(self.img_dir, self.img_list[batch_start:limit])
                X = np.stack([X,X,X], axis=-1)  # transfer learning requires 3 CH.
                
                Y = self._load_img(self.mask_dir, self.mask_list[batch_start:limit])
                Y = np.stack([Y], axis=-1)

                yield(X,Y) # output the X and Y in batch size

                batch_start += self.batch_size 
                batch_end += self.batch_size
                

# for vanilla version: input: img[t, W, H]
class dataGenerator_vanilla:
    def __init__(self, img_dir, img_list, mask_dir, mask_list, batch_size, augment):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list
        self.batch_size = batch_size
        self.aug = augment
        
    def _load_img(self,img_dir, img_list):

        images = []

        for i, image_name in enumerate(img_list):
            # could add pre-process here 
            if (image_name.split('.')[1] == 'npy'):

                image = np.load(img_dir + image_name)  # load npy if data type is correct
                # print(image.min(), image.max())

                if image.max() != 1:
                    # image = image/255  
                    image = norm_01(image) # rescale into [0, 1]

                images.append(image)
            else:
                print('illegal data format')

        images = np.array(images)  # convert into array

        return images.astype('float64')  # for NN
    
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

        L = len(self.img_list)

        # keras require generator to be infinite, so we use while true
        while True:

            batch_start = 0
            batch_end = self.batch_size

            while batch_start < L:

                limit = min(batch_end, L) # 考虑最后一个batch分割不完整的情况
                
                X = self._load_img(self.img_dir, self.img_list[batch_start:limit])
                Y = self._load_img(self.mask_dir, self.mask_list[batch_start:limit])
                
                if self.aug == True:
                    X, Y = self._aug_img(X, Y)
                    Y = Y.astype('float64')
                    
                yield(X,Y) # output the X and Y in batch size

                batch_start += self.batch_size # 都往后挪一个batch
                batch_end += self.batch_size