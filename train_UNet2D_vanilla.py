# train for 2D UNet. Vanilla version
# input: img[NUM, W, H]; msk[NUM, W, H]
# output: predict [NUM, W, H, CH=1]

import os 
import numpy as np
from dataGenerator import imageLoader
import keras
import matplotlib.pyplot as plt
import glob
import random

import config_vanilla

import tensorflow.keras as K
import segmentation_models as sm

from models.simple2DUnet_256 import *

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

def UNet(config):
    
    LR = config.learning_rate
    
    if config.optimizer == 'Adam':
        optim = K.optimizers.Adam(LR)
    else:
        print('specify the optimizer.')
        
    # loss
    BCE_loss = sm.losses.BinaryCELoss()
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()

    total_loss = dice_loss*1 + (1* focal_loss) + BCE_loss*0.005

    # metrics
    metrics = ['accuracy', sm.metrics.IOUScore()]

    # compile the model
    input_shape = (config.input_shape[0], config.input_shape[1], config.input_shape[2])
    model = simple_unet_model(config.input_shape[0], config.input_shape[1], config.input_shape[2])
    
    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

    # print(model.summary)
    print('input shape:', model.input_shape)
    print('output shape:', model.output_shape)
    
    return model

def train(config):
    
    # path define
    
    # training
    train_img_dir = config.data_path + 'train/images/'
    train_msk_dir = config.data_path + 'train/masks/'
    train_img_list = sorted(os.listdir(train_img_dir))  # ensure img and msk paired
    train_msk_list = sorted(os.listdir(train_msk_dir))

    # testing
    test_img_dir = config.data_path + 'test/images/'
    test_msk_dir = config.data_path + 'test/masks/'
    test_img_list = sorted(os.listdir(test_img_dir))
    test_msk_list = sorted(os.listdir(test_msk_dir))

    # validation
    val_img_dir = config.data_path + 'val/images/'
    val_msk_dir = config.data_path + 'val/masks/'
    val_img_list = sorted(os.listdir(val_img_dir))
    val_msk_list = sorted(os.listdir(val_msk_dir))
    
    # generator
    train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                   train_msk_dir, train_msk_list, config.batch_size)

    val_img_datagen = imageLoader(val_img_dir, val_img_list,
                                 val_msk_dir, val_msk_list, config.batch_size)
    
    
    # fetch model
    my_model = UNet(config)
    
    # define the training params
    ## define the neptune
    if config.neptune_document:

        run = neptune.init(
            project="leeleeroy/digitalConfocal-zebrafish",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YjVjOGVmZi04MjA4LTQ4N2QtOWIzYy05M2YyZWI1NzY3MmEifQ==",
            name = "UNet2D_256_vanilla",
        ) # necessary credentials, the name could be used to reproduce the results 

        # for callbacks in training
        neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')  # neptune for the training process

        # neptune document the hyper param.
        PARAMS = {
                  "optimizer": {"learning_rate": config.learning_rate, "optimizer": config.optimizers},
                  'epochs': config.epochs,
                  'batch_size':confg.batch_size}
        
        # log hyper-parameters
        run['hyper-parameters'] = PARAMS
        run["sys/tags"].add(["vanilla", "epochs:" + str(config.epochs)])
        
    ## steps and epochs
    steps_per_epoch = len(train_img_list) // config.batch_size
    val_steps_per_epoch = len(val_img_list) // config.batch_size

    # print('train steps/epoch', steps_per_epoch)
    # print('val steps/epoch', steps_per_epoch)

    # call backs for documentation
    if config.neptune_document:
        callbacks = [
            # k.callbacks.EarlyStopping(patience=15, monitor='val_loss'),
            neptune_cbk, 
            k.callbacks.TensorBoard(log_dir = config.tensorboard_path)  # save in new folder in hemera. Also update in neptune
        ]
    else:
        callbacks = [
            # k.callbacks.EarlyStopping(patience=15, monitor='val_loss'),
            k.callbacks.TensorBoard(log_dir = config.tensorboard_path)  
        ]
    
        
    # training
    history = my_model.fit(train_img_datagen,
                   steps_per_epoch=steps_per_epoch,
                   epochs=config.epochs,
                   verbose=1, # ??
                   validation_data=val_img_datagen,
                   validation_steps=val_steps_per_epoch,
                   callbacks=callbacks)
    
    # save model
    SVAED_MODEL_NAME = config.model_path + 'simple2D_256' + '_' + str(config.epochs) + '.hdf5'
    my_model.save(SVAED_MODEL_NAME)
    
    if DOCUMENT:
        run.stop() 
        
def main():
    
    c = config.configuration()
    print(c)

    # train(c, device, list_A, list_B)
    train(c)
    
    print("finishing ...")
    
    
if __name__ == '__main__':
    main()

# python train.py --epochs 5 --neptune_document False --batch_size 2