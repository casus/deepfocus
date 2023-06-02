import os 
import numpy as np
import keras
import matplotlib.pyplot as plt
import glob
import random
import tensorflow as tf
from sklearn.metrics import jaccard_score
from keras.models import load_model
import keras_contrib as Kc
from tensorflow_addons.layers import InstanceNormalization
import natsort as natsorted

# check the status of GPU
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# metrics
def calculate_iou(y_true, y_pred):
    iou = []
    for i in range(y_true.shape[0]):
        temp = jaccard_score(y_true[i], y_pred[i].squeeze(),average="micro")
        iou.append(temp)
        
    iou = np.asarray(iou)
    return iou.mean()

# reader
def readNpy(DIR, LIST):
    num = len(LIST)
    
    array = []
    
    for i in range(num):
        temp_path = DIR + LIST[i]
        array.append(np.load(temp_path))
        
    return np.asarray(array)

def test_manual(MODEL_PATH, NAME):

    SAVED_MODEL_NAME = MODEL_PATH + NAME   #'best_model_425.h5'
    my_test_model = load_model(SAVED_MODEL_NAME, compile=False, custom_objects={'InstanceNormalization':Kc.layers.InstanceNormalization})

    FT_PATH = '/bigdata/casus/MLID/RuiLi/Data/LM/segStacks/manulSeg/finetune/'

    test_ft_data_dir = FT_PATH + 'test/'
    test_ft_data_list = natsorted(os.listdir(test_ft_data_dir))  # ensure img and msk paired
    print(test_ft_data_list)

    # read in data

    test_ft_data = np.load(test_ft_data_dir + 'test_ft.npz')
    test_ft_img, test_ft_m_msk, test_ft_msk = test_ft_data['img'], test_ft_data['m_mask'], test_ft_data['mask']
    test_ft_img = test_ft_img/255
    print(test_ft_img.shape, test_ft_msk.shape, test_ft_img.max(), np.unique(test_ft_msk))

    # flatten the data
    test_ft_img_flat = test_ft_img.reshape(-1,*test_img.shape[-2:])
    test_ft_m_msk_flat = test_ft_m_msk.reshape(-1,*test_ft_m_msk.shape[-2:])
    print(test_ft_img_flat.shape)

    # prediction on whole test set

    pred_test_all = my_test_model.predict(np.expand_dims(test_ft_img_flat, axis=3))
    print(pred_test_all.shape, pred_test_all.min(), pred_test_all.max())  # non-binary

    # calculate the metrics

    pred_test_all_bi = pred_test_all > 0.2
    iou_test = calculate_iou(test_ft_m_msk_flat, pred_test_all_bi)
    print('finetune dataset:', iou_test)

    return iou_test

def test_vanilla(MODEL_PATH, NAME):
    SAVED_MODEL_NAME = MODEL_PATH + NAME   #'best_model_425.h5'
    my_test_model = load_model(SAVED_MODEL_NAME, compile=False, custom_objects={'InstanceNormalization':Kc.layers.InstanceNormalization})

    TEST_PATH = '/bigdata/casus/MLID/RuiLi/Data/LM/segStacks/manulSeg/preTrain/'

    # test
    test_data_dir = TEST_PATH + 'test/'
    test_data_list = natsorted(os.listdir(test_data_dir))  # ensure img and msk paired
    print(test_data_list)

    # read in data

    test_data = np.load(test_data_dir + 'test_pretrain.npz')
    test_img, test_msk = test_data['img'], test_data['mask']
    test_img = test_img/255

    print(test_img.shape, test_msk.shape, test_img.max(), np.unique(test_msk))

    # flatten the data
    test_img_flat = test_img.reshape(-1,*test_img.shape[-2:])
    test_msk_flat = test_msk.reshape(-1,*test_msk.shape[-2:])

    print(test_img_flat.shape)

    # prediction on whole test set

    pred_test_all = my_test_model.predict(np.expand_dims(test_img_flat, axis=3))
    print(pred_test_all.shape, pred_test_all.min(), pred_test_all.max())  # non-binary

    # calculate the metrics

    pred_test_all_bi = pred_test_all > 0.2
    iou_test = calculate_iou(test_msk_flat, pred_test_all_bi)
    print('vanilla test:', iou_test)

    return iou_test

def main(MODEL_PATH1, MODEL_PATH2, NAME1, NAME2):

    iou_ft = test_vanilla(MODEL_PATH1, NAME1)
    iou_vanilla = test_manual(MODEL_PATH2, NAME2)

    print('iou ft and iou_vanilla:', iou_ft, iou_vanilla)
    print("finishing ...")
    
if __name__ == '__main__':
    main()








