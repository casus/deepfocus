# fine tune for 2D UNet. Vanilla version
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
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

from models.simple2DUnet_256 import *

# define custom loss function
def custom_loss(y_true, y_pred):
    BCE_loss = sm.losses.BinaryCELoss()(y_true, y_pred)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1]))(y_true, y_pred)
    focal_loss = sm.losses.CategoricalFocalLoss()(y_true, y_pred)
    total_loss = dice_loss*0.05 + (1* focal_loss) + BCE_loss*0.95
    return total_loss

def iou(y_true, y_pred):
    return sm.metrics.IOUScore(threshold=0.2)(y_true, y_pred)


def UNet(config):
    
    LR = config.learning_rate
    
    if config.optimizer == 'Adam':
        optim = K.optimizers.Adam(LR)
    else:
        print('specify the optimizer.')
        
    LR = 0.0001
    optim = K.optimizers.Adam(LR)
    wt0, wt1 = 1.26, 0.83

    # Load the model with custom loss function
    MODEL_PATH = './models_weight/'
    # SVAED_MODEL_NAME = MODEL_PATH + 'simple2D_512_update_first_noAug_' + str(EPOCHS) + '.hdf5'
    SAVED_MODEL_NAME = MODEL_PATH + 'simple2D_512_update_first_noAug_func_1000.h5'

    my_model = load_model(SAVED_MODEL_NAME, compile=True, custom_objects={'InstanceNormalization':InstanceNormalization, 'custom_loss': custom_loss, 'iou':iou})

    return my_model

def finetune(config, aug):
    
    AUGMENT = aug

    # define the data path

    DATA_PATH = config.data_path + '/finetune/'  

    # tuneing
    tune_img_dir = DATA_PATH + 'finetune/images/'
    tune_msk_dir = DATA_PATH + 'finetune/m_masks/'
    tune_img_list = sorted(os.listdir(tune_img_dir))  # ensure img and msk paired
    tune_msk_list = sorted(os.listdir(tune_msk_dir))
    
    # generator

    tune_gen_class = dataGenerator_vanilla(tune_img_dir, tune_img_list,
                                   tune_msk_dir, tune_msk_list, config.batch_size, AUGMENT)

    tune_img_datagen = tune_gen_class.imageLoader()
    
    
    # fetch model
    my_model = UNet(config)
    
    # define the neptune
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
    history = my_model.fit(tune_img_datagen,
                   steps_per_epoch=steps_per_epoch,
                   epochs=config.epochs,
                   verbose=1, 
                   callbacks=callbacks)
    
    # save model
    SVAED_MODEL_NAME = config.model_path + 'simple2D_256' + '_finetune' + str(config.epochs) + '.hdf5'
    my_model.save(SVAED_MODEL_NAME)
    
    if DOCUMENT:
        run.stop() 
        
def main():
    
    c = config.configuration()
    print(c)

    # train(c, device, list_A, list_B)
    aug = False  # decide the augmentation
    finetune(c, aug)
    
    print("finishing tuning")
    
    
if __name__ == '__main__':
    main()

# python finetune_UNet2D_vanilla.py --epochs 5 --neptune_document False --batch_size 2