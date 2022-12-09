# custom data generator without data augmentation
# for transfer models. image should be 3 channels

import os
import numpy as np
from skimage.transform import resize

# load images
def load_img(img_dir, img_list):
    
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

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    
    L = len(img_list)
    
    # keras require generator to be infinite, so we use while true
    while True:
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            
            limit = min(batch_end, L) # 考虑最后一个batch分割不完整的情况
            
            # X = load_img(img_list[batch_start:limit])
            # Y = load_img(mask_list[batch_start:limit])
            
            X = load_img(img_dir, img_list[batch_start:limit])
            X = np.stack([X,X,X], axis=-1)  # transfer learning requires 3 CH.
            
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            Y = np.stack([Y], axis=-1)
            
            yield(X,Y) # output the X and Y in batch size
            
            batch_start += batch_size # 都往后挪一个batch
            batch_end += batch_size