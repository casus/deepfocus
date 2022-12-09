# simple 2D unet from the previous 2D unet
# change the 2d conv into 2D conv
# add instance norm inside; deeper the model

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from keras.metrics import MeanIoU
from tensorflow_addons.layers import InstanceNormalization

kernel_initializer =  'he_uniform' #Try others if you want

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs before into [0,1] beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = InstanceNormalization()(c1)
    print('c1 shape', c1.shape)
    # c1 = Dropout(0.1)(c1)  # BN + ReLU, don't need dropout anymore
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    c1 = InstanceNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    # c2 = Dropout(0.1)(c2)
    c2 = InstanceNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    c2 = InstanceNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = InstanceNormalization()(c3)
    # c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    c3 = InstanceNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = InstanceNormalization()(c4)
    # c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    c4 = InstanceNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = InstanceNormalization()(c5)
    # c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    c5 = InstanceNormalization()(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)
    # print('p5:', p5.shape)
    
    # expand one layer, deep the network
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p5)
    c6 = InstanceNormalization()(c6)
    # c6 = Dropout(0.3)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
    
    c6_hidden = InstanceNormalization()(c6)
    # print('before bottle c6:', c6_hidden.shape)
    
    ##################
    
    # Expansive path 
    # u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6_hidden)
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6_hidden) 
    # print('after bottle:', u7.shape)
    print('conc shape:', u7.shape, c5.shape)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = InstanceNormalization()(c7)
    # c6 = Dropout(0.2)(c6)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
    c7 = InstanceNormalization()(c7)
     
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    # c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    # c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u10)
    # c10 = Dropout(0.1)(c10)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c10)

    u11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c1])
    c11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u11)
    # c11 = Dropout(0.1)(c11)
    c11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c11)
    
    
    # last layer. change the sigmoid
    # outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)  # multi-classes segmentation, for binary use sigmoid
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)
    # outputs = Conv2D(num_classes, (1, 1))(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    model.summary()
    
    return model

# initialize the 2D unet
model = simple_unet_model(256, 256, 1)  # 3 CH
print(model.input_shape)
print(model.output_shape)