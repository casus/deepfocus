# data preparation for zebra fish. data reading, scaling, croping, adjusting data format and split into train/val/test

# data loading 
# small data need to preprocess manually

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import pandas as pd
import random
import splitfolders
import argparse

from skimage.transform import resize


def subShow(IMG1, IMG2):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(IMG1, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(IMG2, cmap='gray')
    plt.show()

# rescale the images

def rescaleStack(imageStack, MIN, MAX):

    ImageScale = []
    
    for stack in range(imageStack.shape[0]):
        temp = imageStack[stack,...]
        tempScale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
        # print(stack, tempScale.min(), tempScale.max())
        ImageScale.append(tempScale.astype('int'))
    
    return np.asarray(ImageScale)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default = '/bigdata/casus/MLID/RuiLi/Data/LM/zebrafish_partial_15/', help='path to the data')
    parser.add_argument('--intermediate_dir', default = './data/zebra15/preData', help='pre-processed data')
    parser.add_argument('--inputData_dir', default = './data/zebra15/inputData/', help='input data dir for training')

    parser.add_argument('--crop_pixels', default = 176, help='crop the image background')
    
    parser.add_argument('--rescale_img', default=[0, 255], help='rescale the gray images')
    parser.add_argument('--rescale_msk', default=[0, 255], help='rescale the mask')
    
    args = parser.parse_args()
    
    # check the data dir
    if not os.path.isdir(args.intermediate_dir):
        os.mkdir(args.intermediate_dir)
        
    if not os.path.isdir(args.inputData_dir):
        os.mkdir(args.inputData_dir)
        
    # pre data dir and sub-folder    
    if not os.path.isdir(args.intermediate_dir + '/images/'):
        os.mkdir(args.intermediate_dir + '/images/')
        
    if not os.path.isdir(args.intermediate_dir + '/masks/'):
        os.mkdir(args.intermediate_dir + '/masks/')
    
    # intermdiate data save path
    SAVED_PATH = args.intermediate_dir + '/'
    
    # original size
    Mask = np.load(args.data_dir + 'biMasks15.npy')
    Image = np.load(args.data_dir + 'rawGray15.npy')
    
    # pre-resized data
    # Mask = np.load(args.data_dir + 'biMasks15_256.npy')
    # Image = np.load(args.data_dir + 'rawGray15_256.npy')
    
    # # crop pics
    Mask = Mask[...,args.crop_pixels:(args.crop_pixels+Mask.shape[2])]  
    Image = Image[...,args.crop_pixels:(args.crop_pixels+Image.shape[2])]
    
    # # resize the pics. need for big image
    # numImage = Image.shape[0]
    Image = resize(Image, (Image.shape[0],Image.shape[1],256, 256), anti_aliasing=False)
    Mask = resize(Mask, (Mask.shape[0],Mask.shape[1],256, 256), anti_aliasing=False)
    
    # rescale
    if Image.max() != 255:
        print('rescale the image')
        Image = rescaleStack(Image, args.rescale_img[0], args.rescale_img[1])
    else:
        print('legal image input uint8')

    if Mask.max() != 255:
        print('rescale mask')
        Mask = rescaleStack(Mask, args.rescale_msk[0], args.rescale_msk[1])
    else:
        print('legal mask input uint8')

    # print('image range:', Image.min(), Image.max())
    # print('Mask range:', Mask.min(), Mask.max())
    
    # re-binary the mask
    Mask = (Mask > 0.2).astype('int')
    
    # flatten
    Image = Image.reshape(-1,*Image.shape[-2:])
    Mask = Mask.reshape(-1,*Mask.shape[-2:])
    
    # save intermeidate data
    
    for slice in range(Image.shape[0]):

        temp_img = Image[slice,...]
        np.save(SAVED_PATH + 'images/image_'+str(slice)+'.npy', temp_img)

        temp_msk = Mask[slice,...]
        # print('mask', temp_msk.min(), temp_msk.max())
        np.save(SAVED_PATH + 'masks/mask_'+str(slice)+'.npy', temp_msk)
    
    
    # split into the train/val/test
    
    splitfolders.ratio(SAVED_PATH, output=args.inputData_dir, seed=42, ratio=(.8, .1, .1), group_prefix=None) # default values
    

        