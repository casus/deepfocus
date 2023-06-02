# train for 2D UNet. Vanilla version
# input: img[NUM, W, H]; msk[NUM, W, H]
# output: predict [NUM, W, H, CH=1]

import os 
import numpy as np
import keras
import matplotlib.pyplot as plt
import glob
import random
import config_vanilla
import tensorflow.keras as K
import segmentation_models as sm
from natsort import natsorted
import tensorflow.keras as k
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from models.simple2DUnet_512 import *
from utils import *

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

def train(config, aug):
    
    AUGMENT = aug
    
    # training
    train_data_dir = config.data_path + 'train/'
    train_data_list = natsorted(os.listdir(train_data_dir))  # ensure img and msk paired

    # validation
    val_data_dir = config.data_path + 'val/'
    val_data_list = natsorted(os.listdir(val_data_dir))  # ensure img and msk paired
    
    # generator
    # tranining
    train_gen_class = dataGenerator_allStack(train_data_dir, train_data_list,config.batch_size, augment=AUGMENT)
    train_img_datagen = train_gen_class.imageLoader()

    # validation
    val_gen_class = dataGenerator_allStack(val_data_dir, val_data_list,config.batch_size, augment=AUGMENT)
    val_img_datagen = val_gen_class.imageLoader()
    
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
        
    # steps and epochs
    train_data, val_data = np.load(train_data_dir + 'train_pretrain.npz'), np.load(val_data_dir + 'val_pretrain.npz')
    train_msk_all, val_msk_all = train_data['mask'], val_data['mask']

    steps_per_epoch = train_msk_all.shape[0] // config.batch_size
    val_steps_per_epoch = val_msk_all.shape[0] // config.batch_size

    # checkpoints
    # set the checkpoint    
    filepath = "./models_weight/checkPoints--{epoch:02d}-{val_iou:.2f}.h5"  
    checkpoint_callback = ModelCheckpoint(filepath=filepath,monitor='val_iou',
                                        save_freq='epoch',period=20)
    early_stopping_callback = EarlyStopping(monitor='val_iou', patience=10)  # val_iou -> iou

    # call backs for documentation
    if config.neptune_document:
        callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            neptune_cbk, 
            k.callbacks.TensorBoard(log_dir = config.tensorboard_path)  # save in new folder in hemera. Also update in neptune
        ]
    else:
        callbacks = [
            early_stopping_callback,
            checkpoint_callback,
            k.callbacks.TensorBoard(log_dir = config.tensorboard_path)  
        ]
    
    patience = config.patience  # Number of epochs with no improvement
    best_loss = float('inf')  # Initialize best validation loss
    counter = 0  # Counter for epochs with no improvement
    best_weights = None  # Variable to store the best weights

    for step in range(config.epochs):
        # Training step
        img, msk = train_img_datagen.__next__()
        img = img / 255  # Normalize image [0, 1]
        
        loss = model.train_on_batch(img, msk.astype('float64'))  # Batch size = 1
        
        print("Training step:", step, "Loss:", loss[0], "iou:", loss[1])
        
        # Validation step
        if step % 25 == 0:
            val_img, val_msk = val_img_datagen.__next__()
            val_img = val_img / 255  # Normalize validation image [0, 1]
            
            val_loss = model.test_on_batch(val_img, val_msk.astype('float64'))  # Batch size = 1
            val_loss = np.asarray(val_loss)
            
            print("Validation step:", step, "Loss:", val_loss[0], "iou:", val_loss[1])
            
            # Check early stopping criteria
            if val_loss[0] < best_loss:
                best_loss = val_loss[0]
                counter = 0
                # Save the best weights
                best_weights = model.get_weights()
            else:
                counter += 1
            
            if counter >= patience:
                print("Early stopping triggered. Restoring best weights.")
                model.save('./models_weight/' + 'best_model_' + str(step) + '.h5')
                break
    
    if config.neptune_document:
        run.stop() 
        
def main():
    
    c = config.configuration()
    print(c)
    aug = False  # decide the augmentation
    train(c, aug)
    
    print("finishing ...")
    
if __name__ == '__main__':
    main()

# python train.py --epochs 5 --neptune_document False --batch_size 2