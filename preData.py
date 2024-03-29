# data preparation for zebra fish. data reading, scaling, croping, adjusting data format and split into train/val/test

# data loading 
# all the 69 stacks of data

import os
import numpy as np
from scipy import ndimage
import random
import argparse
from skimage.transform import resize
import matplotlib.pyplot as plt

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
    parser.add_argument('--data_dir', default = 'xxxxxx/', help='path to the data')
    parser.add_argument('--inputData_dir', default = '/xxxxxx/', help='final input data dir for training')

    parser.add_argument('--crop_pixels', default = xxx, help='crop the image background')
    
    parser.add_argument('--rescale_img', default=[0, 255], help='rescale the gray images')
    parser.add_argument('--rescale_msk', default=[0, 1], help='rescale the mask')
    parser.add_argument('--split_ratio', default=[.8, .1, .1], help='ratio for train/val/test')
    
    args = parser.parse_args()
        
    if not os.path.isdir(args.inputData_dir):
        os.mkdir(args.inputData_dir)
        
    # intermdiate data save path
    SAVED_PATH = args.inputData_dir + '/'
    
    # original images
    raw_data = np.load(args.data_dir + 'xxxx.npz')  # whole stacks
    Mask, Image = raw_data['mask'], raw_data['img']
    
    # crop pics
    Mask = Mask[...,args.crop_pixels:(args.crop_pixels+Mask.shape[2])].astype('bool')    
    Image = Image[...,args.crop_pixels:(args.crop_pixels+Image.shape[2])]
    
    # resize the pics
    Image = resize(Image, (Image.shape[0],Image.shape[1],256, 256), anti_aliasing=False)
    Mask = resize(Mask, (Mask.shape[0],Mask.shape[1],256, 256), anti_aliasing=False)
    Mask = Mask.astype('int')
    
    # rescale
    if Image.max() != 255:
        print('rescale the image')
        Image = rescaleStack(Image, args.rescale_img[0], args.rescale_img[1])
    else:
        print('legal image input uint8')

    if Mask.max() != 1:
        print('rescale mask')
        Mask = rescaleStack(Mask, args.rescale_msk[0], args.rescale_msk[1])
    else:
        print('legal mask input binary')

    # split train/val/test
    ratio = np.asarray(args.split_ratio)
    ratio = (ratio* Image.shape[0]).round().astype('int')
    print('split stack nums:', ratio)

    train_img, val_img, test_img = Image[:ratio[0]], Image[ratio[0]:ratio[0]+ratio[1]], Image[ratio[0]+ratio[1]:ratio[0]+ratio[1]+ratio[2]]
    train_msk, val_msk, test_msk = Mask[:ratio[0]], Mask[ratio[0]:ratio[0]+ratio[1]], Mask[ratio[0]+ratio[1]:ratio[0]+ratio[1]+ratio[2]]

    # flatten train and val
    train_img_flat, val_img_flat = train_img.reshape(-1,*train_img.shape[-2:]), val_img.reshape(-1,*val_img.shape[-2:])
    train_msk_flat, val_msk_flat = train_msk.reshape(-1,*train_msk.shape[-2:]), val_msk.reshape(-1,*val_msk.shape[-2:])

    # save 
    np.savez(args.inputData_dir+'train_pretrain.npz', img=train_img_flat, mask=train_msk_flat)
    np.savez(args.inputData_dir+'val_pretrain.npz', img=val_img_flat, mask=val_msk_flat)
    np.savez(args.inputData_dir+'test_pretrain.npz', img=test_img, mask=test_msk)

    print('data saved at:', args.inputData_dir)