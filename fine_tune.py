import tensorflow as tf
import os 
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted
from utils import *
import tensorflow.keras as K
import segmentation_models as sm
from keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

# define custom loss function
def custom_loss(y_true, y_pred):
    BCE_loss = sm.losses.BinaryCELoss()(y_true, y_pred)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1]))(y_true, y_pred)
    focal_loss = sm.losses.CategoricalFocalLoss()(y_true, y_pred)
    total_loss = dice_loss*0.05 + (1* focal_loss) + BCE_loss*0.95
    return total_loss

def iou(y_true, y_pred):
    return sm.metrics.IOUScore(threshold=0.2)(y_true, y_pred)


def fine_tune(DATA_PATH, EPOCHS=100, DOCUMENT=False, batch_size=16, AUGMENT=True):

    if DOCUMENT:

        run = neptune.init(
            project="xxxx",
            api_token="xxxx",
            name = "UNet2D_fineTune",
        ) # necessary credentials, the name could be used to reproduce the results 

        neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')  # neptune for the training process
        
        # neptune document the hyper param.
        PARAMS = {
                "optimizer": {"learning_rate": 0.0001, "optimizer": "Adam"},
                'epochs': EPOCHS,
                'batch_size':batch_size}

        # log hyper-parameters
        run['hyper-parameters'] = PARAMS
        run["sys/tags"].add(["vanilla", "epochs:" + str(EPOCHS), 'deep512','allData'])

    # data 
    train_data_dir = DATA_PATH + 'train/'
    train_data_list = natsorted(os.listdir(train_data_dir))  # ensure img and msk paired
    val_data_dir = DATA_PATH + 'val/'
    val_data_list = natsorted(os.listdir(val_data_dir))  

    train_gen_class = dataGenerator_allStack_ft(train_data_dir, train_data_list,batch_size, augment=AUGMENT)
    train_img_datagen = train_gen_class.imageLoader()
    val_gen_class = dataGenerator_allStack_ft(val_data_dir, val_data_list,batch_size, augment=AUGMENT)
    val_img_datagen = val_gen_class.imageLoader()

    # steps and epochs
    train_data, val_data = np.load(train_data_dir + 'train_ft.npz'), np.load(val_data_dir + 'val_ft.npz')
    train_msk_all, val_msk_all = train_data['mask'], val_data['mask']

    steps_per_epoch = train_msk_all.shape[0] // batch_size
    val_steps_per_epoch = val_msk_all.shape[0] // batch_size

    # call backs for documentation
    filepath = "./models_weight/saved-ft-model-early--{epoch:02d}-{val_iou:.2f}.h5"  # save the weights
    checkpoint_callback = ModelCheckpoint(filepath=filepath,monitor='val_iou',
                                        save_freq='epoch',period=20)

    early_stopping_callback = EarlyStopping(monitor='val_iou', patience=10)  # val_iou -> iou

    if DOCUMENT:
        callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            neptune_cbk, 
            K.callbacks.TensorBoard(log_dir = './tensorboard/UNet2D'),  # save in new folder in hemera. Also update in neptune
            
        ]
    else:
        callbacks = [
            early_stopping_callback,
            checkpoint_callback,
            K.callbacks.TensorBoard(log_dir = './tensorboard/UNet2D')
        ]



    LR = 0.0001
    optim = K.optimizers.Adam(LR)
    wt0, wt1 = 1.26, 0.83

    # Load the model with custom loss function
    MODEL_PATH = './models_weight/'
    SAVED_MODEL_NAME = MODEL_PATH + 'xxx.h5'

    my_ft_model = load_model(SAVED_MODEL_NAME, compile=True, custom_objects={'InstanceNormalization':InstanceNormalization, 'custom_loss': custom_loss, 'iou':iou})

    NUM_STEPS = EPOCHS

    # EarlyStopping parameters
    patience = 20  # Number of epochs with no improvement
    best_loss = float('inf')  # Initialize best validation loss
    counter = 0  # Counter for epochs with no improvement
    best_weights = None  # Variable to store the best weights

    for step in range(NUM_STEPS):
        # Training step
        img, _, msk = train_img_datagen.__next__()  # take only m_mask
        img = img / 255  # Normalize image [0, 1]
        
        loss = my_ft_model.train_on_batch(img, msk.astype('float64'))  # Batch size = 1
        
        print("Training step:", step, "Loss:", loss[0], "iou:", loss[1])
        
        # Validation step
        if step % 25 == 0:
            val_img, _, val_msk = val_img_datagen.__next__()
            val_img = val_img / 255  # Normalize validation image [0, 1]
            
            val_loss = my_ft_model.test_on_batch(val_img, val_msk.astype('float64'))  # Batch size = 1
            val_loss = np.asarray(val_loss)
            
            print("Validation step:", step, "Loss:", val_loss[0], "iou:", val_loss[1])
            
            # Check early stopping criteria
            if val_loss[0] < best_loss:
                best_loss = val_loss[0]
                counter = 0
                # Save the best weights
                best_weights = my_ft_model.get_weights()
            else:
                counter += 1
            
            if counter >= patience:
                print("Early stopping triggered. Restoring best weights.")
                my_ft_model.save('./models_weight/' + 'best_ft_model_' + str(step) + '.h5')
                break
        
        # Save my_ft_model weights
        if step % 100 == 0:
            print('check points:', step)
            my_ft_model.save('./models_weight/' + 'ft_model_' + str(step) + '.h5')

def main():
    
    DATA_PATH = 'xxx/'  
    fine_tune(DATA_PATH)
    
if __name__ == '__main__':
    main()

