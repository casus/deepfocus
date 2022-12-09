#!/usr/bin/env python
# coding: utf-8

# #### cycleGAN to transform between masks and widefield microscope images

# #### input: images: (t, H, W, C) | masks: (t, H, W, C). Outputs: (t, H, W, C). C=3

# In[20]:


# detect the GPU status

import tensorflow as tf

print(tf.__version__)

from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()

[print(x) for x in local_device_protos if x.device_type == 'GPU']


# In[21]:


# define if the documenting process should go on

DOCUMENT = False
TRAIN = 600 # training epochs num


# In[22]:


# neptune document

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

if DOCUMENT:

    run = neptune.init(
        project="leeleeroy/LM-2D-GAN",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YjVjOGVmZi04MjA4LTQ4N2QtOWIzYy05M2YyZWI1NzY3MmEifQ==",
        name = "CycleGAN_vanilla",
    ) # necessary credentials, the name could be used to reproduce the results 

    # for callbacks in training

    neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')  # neptune for the training process
    
    # neptune document the hyper param.

    PARAMS = {
              "optimizer": {"learning_rate": 0.001, "beta_1":0.9,"optimizer": "Adam"},
              'epochs': TRAIN,
              'batch_size':8}

    # log hyper-parameters
    run['hyper-parameters'] = PARAMS
    run["sys/tags"].add(["vanilla", "val", "binary", "epochs:300"])


# #### Load in the data

# In[23]:


# data loading 

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import pandas as pd
import random


# In[24]:


# visualization for two images

def subShow(IMG1, IMG2):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(IMG1, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(IMG2, cmap='gray')
    plt.show()


# In[25]:


# # preprocess the data: resize, adding channel, and rescale into [a1, a2]
# # imgStack: mask with 0,1

# from skimage.transform import resize

# def pre_process(imgStack, maskStack, CHANNEL, bound = [a1,a2]):
    


# In[26]:


PATH = '/bigdata/casus/MLID/RuiLi/Data/LM/zebrafish_partial_15/'

Mask = np.load(PATH + 'biMasks15.npy')
IMG = np.load(PATH + 'rawGray15.npy')

Mask = Mask.reshape(-1, 1040, 1392)  # flatten into images 
IMG = IMG.reshape(-1, 1040, 1392)

Mask = Mask[...,176:(176+Mask.shape[1])]  # crop for later scaling
IMG = IMG[...,176:(176+IMG.shape[1])]

print('Mask info: ', Mask.shape, Mask.dtype)
print('Image info: ', IMG.shape, IMG.dtype)


# In[27]:


# resize the images

from skimage.transform import resize

SIZE = [256, 256]
totalIMG = Mask.shape[0]
numIMG = 250

smallIMG = resize(IMG[:numIMG,...], (numIMG,SIZE[0],SIZE[1]), anti_aliasing=True)
smallIMG = np.interp(smallIMG, (smallIMG.min(), smallIMG.max()), (-1, 1))  # rescale the img into [-1, 1] for cycleGAN

smallMask = resize(Mask[:numIMG,...].astype(bool), (numIMG,SIZE[0],SIZE[1]), anti_aliasing=False)
smallMask = smallMask.astype(int)
smallMask = np.interp(smallMask, (smallMask.min(), smallMask.max()), (-1, 1))


# In[28]:


# sanity check

NUM = 100

subShow(Mask[NUM,...], IMG[NUM,...])

subShow(smallMask[NUM,...], smallIMG[NUM,...])


# In[29]:


# patchify the images
from patchify import patchify, unpatchify

def rawPatch(imageStack,patchPara):
    all_img_patches = []

    for img in range(imageStack.shape[0]):
        large_image = imageStack[img]

        patches_img = patchify(large_image, (patchPara['x'],patchPara['y']), step=patchPara['step'])  # no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):

                single_patch_img = patches_img[i,j,:,:]
                # transform the image if the type is not correct
                if single_patch_img.dtype == 'uint8':
                    single_patch_img = (single_patch_img.astype('float32')) / 255.  # remember to standarize into 0-1
                    
                all_img_patches.append(single_patch_img)
    
    return all_img_patches, patches_img.shape


# In[30]:


# sanity check for the resized data

subShow(smallIMG[0,...], smallMask[0,...])

print('img:',smallIMG.shape, smallIMG.dtype)
print('mask:',smallMask.shape, smallMask.dtype)

print('img range:', np.max(smallIMG), np.min(smallIMG))
print('mask range:', np.max(smallMask), np.min(smallMask))


# In[31]:


# preporcessing the data into patches and change into 3 channels

# train dataset
patchPara = {'x': 256, 'y': 256, 'step':256}

X_patches, _ =  rawPatch(smallIMG, patchPara); X_patches = np.stack((X_patches,)*3, axis=-1)
Y_masks, _ = rawPatch(smallMask, patchPara); Y_masks = np.stack((Y_masks,)*3, axis=-1) #Y_masks = np.expand_dims(Y_masks, -1)


# In[32]:


# sanity check on one layer
print('img:', X_patches.shape)
print('masks:', Y_masks.shape)

subShow(X_patches[0,:,:,0], Y_masks[0,...])


# In[33]:


# check the data properties

print('patches shape:',X_patches.shape, X_patches.dtype)
print('mask shape:',Y_masks.shape, Y_masks.dtype)
print(np.max(Y_masks[0,...]), np.min(Y_masks[0,...]))
print(np.max(X_patches[0,...]), np.min(X_patches[0,...]))


# In[34]:


#  sanity check for the mask and images

startNum = 100
n_samples = 4

for i in range(n_samples):
    plt.subplot(2, n_samples, 1+i)
    plt.axis('off')
    plt.imshow(X_patches[int(i+startNum),:,:,0], cmap='gray')  # only visualize one channel
    
for i in range(n_samples):
    plt.subplot(2, n_samples, 1+n_samples+i)
    plt.axis('off')
    plt.imshow(Y_masks[int(i+startNum)], cmap='gray')    
plt.show()


# #### Prepare the model

# In[35]:


from random import random
import numpy as np
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
import tensorflow.keras as k

# use instance normalization as suggested in paper
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import matplotlib.pyplot as plt


# #### Discriminator. 70x70 patch GAN

# In[36]:


# C64-C128-C256-C512
# After last layer, conv to 1-dimension then go through sigmoid
# axis of instancenorm is '-1', ensure features are normalized per feature map

def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = k.layers.Input(shape=image_shape)
    
    #C64: 4x4 kernel, strides 2x2
    d = k.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    d = k.layers.LeakyReLU(alpha=0.2)(d)
    # C128: 4x4 kernel, strides 2x2
    d = k.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d) # first norm then activate
    d = k.layers.LeakyReLU(alpha=0.2)(d)
    # C256: 4x4 kernel, strides 2x2
    d = k.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = k.layers.LeakyReLU(alpha=0.2)(d)
    # C512: 4x4 kernel, stride 2x2
    # DIY layer,not in original paper
    d = k.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = k.layers.LeakyReLU(alpha=0.2)(d)
    
    # second last layer. kernel 4x4, but stride 1x1
    d = k.layers.Conv2D(512, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = k.layers.LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = k.layers.Conv2D(1, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(d)
    
    # define model
    model = k.models.Model(in_image, patch_out)
    # compile the model
    # batch size is 1, Adam as opt.
    # loss of D is weighted by 50% of each update. This slows down D's change to G during training
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='mse', optimizer=opt, loss_weights=[0.5]) # !!! loss_weights is plural
    return model
    


# #### Generator. based on resnet

# In[37]:


# residual blocks contain two 3x3 Conv with same number of filters in layers

# to release the gradient vanishing and exploding
def resnet_block(n_filters, input_layer):
    # weight init
    init = RandomNormal(stddev=0.02)
    # first conv layer
    g = k.layers.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
#     g = k.layers.ReLU()(g)
    g = k.layers.Activation('relu')(g)  # ??? to layers.ReLU. 只有leaky直接调用，其他的用activation调用
    # second conv layer
    g = k.layers.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # merge channels with input-layer
    g = k.layers.Concatenate()([g, input_layer])
    return g


# In[38]:


# define G model: unet same structure

# c7s1-k: 7x7 Conv -stride 1 -Instancenorm-ReLU -k filters
# dk: 3x3 conv -stride 2 -Instancenorm-ReLU -k filters
# Rk: residual block that contains two 3x3 conv layers
# uk: 3x3 fractional~strided~conv -stride 1/2 -Instancenorm -ReLU -k filters

# two possible structures:
# with 6 res-blocks: c7s1-64, d128, d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
# with 9 res-blocks: c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128, u64,c7s1-3

def define_generator(image_shape, n_resnet=9):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = k.layers.Input(shape=image_shape)
    
    # c7s1-64
    g = k.layers.Conv2D(64, (7,7), strides=(1,1), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = k.layers.Activation('relu')(g)
    # d128
    g = k.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = k.layers.Activation('relu')(g)
    # d256
    g = k.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = k.layers.Activation('relu')(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)  # !!! generate the resnet
    # u128
    g = k.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = k.layers.Activation('relu')(g)
    # u64
    g = k.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = k.layers.Activation('relu')(g)
    # c7s1-3
    g = k.layers.Conv2D(3, (7,7), strides=(1,1), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = k.layers.Activation('tanh')(g)
    # define the model
    model = k.models.Model(in_image, out_image)  # generator does not compile
    return model


# In[39]:


# define a composite model to update generator wuth adversarial and cycle loss
# only use to train generator
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    # use this to train both generators. But one at a time
    # trained G is tranable, others are constant
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False
    
    # adversarial loss
    input_gen = k.layers.Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    ouput_d = d_model(gen1_out)
    # identity loss
    input_id = k.layers.Input(shape=image_shape)
    output_id = g_model_1(input_id)
    # cycle-loss forward
    output_f = g_model_2(gen1_out)
    # cycle-loss backward
    gen2_out = g_model_2(input_id)  # ???
    output_b = g_model_1(gen2_out)
    
    # define the model graph
    model = k.models.Model([input_gen, input_id], [ouput_d, output_id, output_f, output_b])
    
    # compile the model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1,5,10,10], optimizer=opt) # hyper param from paper
    return model


# #### processing images

# In[40]:


# load and prepare traning images
# in cycleGAN scale between -1 and 1, last layer is tanh activation

def load_real_samples(filename):
    # load the dataset
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1-127.5)/127.5
    X2 = (X2-127.5)/127.5
    return [X1, X2]

# D needs fake and real images
# select batch of samples, return images and target.
# real images the label is '1'
def generate_real_samples(dataset, n_samples, patch_shape):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    # generate the 'real' class label
    y = np.ones((n_samples, patch_shape, patch_shape, 1)) # in th same size of one channel
    return X, y

# fake images with label '0'
def generate_fake_samples(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# save model
def save_models(PATH, step, g_model_AtoB, g_model_BtoA):
    # save the first generator models
    filename1 = PATH + 'g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    # save the second generator model
    filename2 = PATH + 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))  


# In[41]:


# predict images with save model, plot input and output images
def summarize_performance(step, g_model, trainX, name, n_samples=5):
    # sample input images
    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    # generate translated images
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    # scale pixels from [-1,1] to [0,1]
    X_in = (X_in + 1)/2
    X_out = (X_out + 1)/2
    # plot real images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1+i)
        plt.axis('off')
        plt.imshow(X_in[i])
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1+i+n_samples)
        plt.axis('off')
        plt.imshow(X_out[i])
    # save plot
    filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
    plt.savefig('./cycleGAN/IMG/' + filename1)
    plt.close()
        


# In[42]:


# from random import random
# test = random()
# print(test)


# In[43]:


# update fake images pool to avoid model oscillation
# update D using a history of generated images rather than latest generators
# image buffer is 50

from random import random

def update_image_pool(pool, images, max_size=20):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
        # elif random.random() < 0.5:  # weird, must indicate the package name
            # use images, but don't add it into pool
            selected.append(image)
        else:
            # replace exiting images and use replaced 
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    out = np.asarray(selected)  # transfer into array
    return out


# #### training process

# In[44]:


# train cyclegan
def train(savePATH, d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=1):
    # save the training information
    loss_all = []
    # training hyper param
    n_epochs, n_batch = epochs, 1 # batch_size is fixed into 1
    # output square shape of D
    n_patch = d_model_A.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # image pool for fake images
    poolA, poolB = list(), list()
    # number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # training iterations
    n_steps = bat_per_epo * n_epochs
    
    # enumerate epochs
    for i in range(n_steps):
        # for every iteration/ batch
        # sample real images from both domain
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        
        # generate fake images for both (A2B, B2A)
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        # update the fake images in the pool as buffer with 50 images
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        
        # update G-B2A via composite model
        # this is the combine model: G1 + G2 + D
        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # update D for A->[real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)  # since batch is 1, cannot split into half for real/fale
        
        # update G-A2B via composite model
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        
        # collect the loss information
        loss_all.append([dA_loss1, dA_loss2, g_loss1, dB_loss1, dB_loss2, g_loss2])
        
        # summarize the performance
        # batch size is 1, iteration is same as dataset
        # if there are 100 images, then 1 epoch will be 100 iterations
        print('Iterations>%d, dA[%.3f, %3.f] dB[%.3f,%.3f] g[%.3f, %.3f]'% (i+1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
        
        # evaluate performance periodically
        if (i+1) % (bat_per_epo * 1) == 0:
            # plot A->B translation
            summarize_performance(i, g_model_AtoB, trainA, 'A2B')
            # plot B->A translation
            summarize_performance(i, g_model_BtoA, trainB, 'B2A')
        
        # save model every 5 batches
        # if (i+1) % (bat_per_epo * 1) == 0:
        if (i+1) % (bat_per_epo * 5) == 0:
            # if batch size(total images)=100, model saved after every 75th * 5 = 375 iter 
            # save_models(i, './cycleGAN/model/' + g_model_AtoB, './cycleGAN/model/' + g_model_BtoA)
            save_models(savePATH, i, g_model_AtoB, g_model_BtoA)
            
    np.save('./cycleGAN/cycleGAN_600.npy', np.asarray(loss_all))


# #### prepare the data

# In[45]:


from os import listdir
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

# the data has been rescaled into [-1, 1]
images = X_patches
masks = Y_masks
X_train, X_val, Y_train, Y_val = train_test_split(images, masks, test_size = 0.25, random_state = 42)


# In[46]:


# sanity check for the data

NUM = 100

subShow(images[NUM,:,:,0], masks[NUM,...])

print(X_train.shape, Y_train.shape)


# In[47]:


# image pre-processing

dataset = [X_train, Y_train]  # single channel

print('processed: ', dataset[0].shape, dataset[1].shape)


# In[48]:


# sanity check
n_samples = 3

for i in range(n_samples):
    plt.subplot(2, n_samples, i+1)
    plt.axis('off')
    plt.imshow(dataset[0][i])
    
for i in range(n_samples):
    plt.subplot(2, n_samples, i+1+n_samples)
    plt.axis('off')
    plt.imshow(dataset[1][i])
plt.show()


# In[49]:


image_shape = dataset[0].shape[1:]


# In[50]:


# define hyper param and intance of model in cycleGAN
image_shape = dataset[0].shape[1:]
# generator A->B
g_model_AtoB = define_generator(image_shape)
# generator B->A
g_model_BtoA = define_generator(image_shape)
# D: A-> [real/ fake]
d_model_A = define_discriminator(image_shape)
# D: B-> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A->B->[real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B->A->[real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)


# #### Training process

# In[ ]:


# training

from datetime import datetime
start1 = datetime.now()

# train model
savePATH = './cycleGAN/model/'
# train(savePATH, d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=1)
train(savePATH, d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=TRAIN)

stop1 = datetime.now()


# In[ ]:


# execustion time
execution_time = stop1 - start1
print('Executed time: ', execution_time)


# # #### test the model

# # In[51]:


# # load model from local path

# import tensorflow
# import tensorflow.keras

# from keras.models import load_model

# from patchify import patchify, unpatchify
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage
# import os
# import pandas as pd
# import random


# # load models. it's saved every 5 epochs
# cust = {'InstanceNormalization': InstanceNormalization}  # !!! the lcoal normalization 
# model_AtoB = load_model('./cycleGAN/model/g_model_AtoB_016830.h5', cust)
# model_BtoA = load_model('./cycleGAN/model/g_model_BtoA_016830.h5', cust)


# # In[52]:


# # select random samples from dataset
# def select_samples(dataset, n_samples):
#     ix = randint(0, dataset.shape[0], n_samples)
#     X = dataset[ix]
#     return X


# # In[53]:


# # plot the image, translation and reconstruction
# def show_plot(imagesX, imagesY1, imagesY2):
#     images = np.vstack((imagesX, imagesY1, imagesY2))
#     titles = ['Real', 'Generated', 'Reconstructed']
#     # scale from [-1,1] to [0,1]
#     images = (images + 1) / 2.0
#     # plot the images
#     for i in range(len(images)):
#         plt.subplot(1, len(images), 1+i)
#         plt.axis('off')
#         plt.imshow(images[i])
#         plt.title(titles[i])
#     plt.show()


# # In[54]:


# # prepare the test datasset for the rest of the images

# testIMG = IMG[numIMG:totalIMG,...]
# testMask = Mask[numIMG:totalIMG,...]

# testIMG = testIMG[...,:testIMG.shape[1]]  # crop for later scaling
# testMask = testMask[...,:testMask.shape[1]]

# print(testIMG.shape)


# # In[55]:


# # pre-process the dataset. resize, adding channel and scale into [-1, 1]

# # resize
# X_test = resize(testIMG, (totalIMG - numIMG,SIZE[0],SIZE[1]), anti_aliasing=True)  # resize the images
# X_test = np.interp(X_test, (X_test.min(), X_test.max()), (-1, 1))

# Y_test = resize(testMask.astype(bool), (totalIMG - numIMG,SIZE[0],SIZE[1]), anti_aliasing=False)
# Y_test = Y_test.astype(int)
# Y_test = np.interp(Y_test, (Y_test.min(), Y_test.max()), (-1, 1))

# # add channel
# X_test, _ =  rawPatch(X_test, patchPara); X_test = np.stack((X_test,)*3, axis=-1)
# Y_test, _ = rawPatch(Y_test, patchPara); Y_test = np.stack((Y_test,)*3, axis=-1) #Y_masks = np.expand_dims(Y_masks, -1)

# print('image:',X_test.shape, np.max(X_test), np.min(X_test))
# print('mask:',Y_test.shape, np.max(Y_test), np.min(Y_test))


# # In[56]:


# # predict the dataset

# A_data = X_test  # image (microscope)
# B_data = Y_test  # masks

# predict_data = [A_data, B_data]


# # In[57]:


# # prediction of the results

# B_generated  = model_AtoB.predict(predict_data[0])  # micro > mask
# A_generated  = model_BtoA.predict(predict_data[1])  # mask > micro


# # In[60]:


# print(B_generated.shape, A_generated.shape, predict_data[0].shape)


# # In[ ]:





# # In[63]:


# # concatenate the images (input, gen, GT), document with neptune

# def concTarSrc(source, gen, target):

#     imagePred = []

#     for i in range(gen.shape[0]):
#         tIMG = source[i,...]  # input
#         tPred = gen[i,...] # prediction
#         tMask = target[i,...] # GT

#         bar = np.ones((tIMG.shape[0], 15))   # lines
#         combTemp = np.concatenate((np.squeeze(tIMG[...,0]), bar, np.squeeze(tPred[...,0]), bar, np.squeeze(tMask[...,0])), axis=1)


#         # upload the test images to neptune
#         # if DOCUMENT:
#         #     # upload the test results into neptune with handle 'description'
#         #     run["test/sample_images"].log(neptune.types.File.as_image(combTemp), name=str(i), description='test images')  

#         imagePred.append(combTemp)

#     imagePred = np.asarray(imagePred)
    
#     return imagePred


# # In[66]:


# A2B_val = concTarSrc(predict_data[0], B_generated, predict_data[1])
# B2A_val = concTarSrc(predict_data[1], A_generated, predict_data[0])
# print(A2B_val.shape, B2A_val.shape)


# # In[68]:


# subShow(A2B_val[0,...], B2A_val[0,...])


# # In[69]:


# for i in range(A2B_val.shape[0]):
#     plt.imshow(A2B_val[i,...], cmap='gray')
#     plt.savefig('./cycleGAN/val/A2B/{}.png'.format(i))


# # In[70]:


# for i in range(B2A_val.shape[0]):
#     plt.imshow(B2A_val[i,...], cmap='gray')
#     plt.savefig('./cycleGAN/val/B2A/{}.png'.format(i))


# # #### upload to neptune

# # #### check for the test dataset

# # In[ ]:


# # 检查当前预测的图像

# print(B_generated.shape, A_generated.shape)

# NUM = 1
# subShow(A_generated[NUM,...], predict_data[1][NUM,...])
# subShow(B_generated[NUM,...], predict_data[0][NUM,...])


# # In[41]:


# # sanity check for some random images

# # plot A->B->A (microscope to masks to microscope)
# A_real = select_samples(A_data, 1)
# B_generated  = model_AtoB.predict(A_real)
# A_reconstructed = model_BtoA.predict(B_generated)
# show_plot(A_real, B_generated, A_reconstructed)

# # plot B->A->B (mask to microscope to mask)
# B_real = select_samples(B_data, 1)
# A_generated  = model_BtoA.predict(B_real)
# B_reconstructed = model_AtoB.predict(A_generated)
# show_plot(B_real, A_generated, B_reconstructed)


# # In[ ]:





# # In[ ]:




