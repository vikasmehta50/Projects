import tifffile as tiff
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
from collections import OrderedDict
import time
import h5py

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model,regularizers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, SeparableConv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Conv3DTranspose, concatenate, AveragePooling3D, UpSampling3D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
#from stacked_models_block_controlled_demo import TrainHistory

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters
#from k_Unet_avg_pool4 import get_model as get_model_ori

# from k_imageGen8_3 import getImage, one_hot_labels, read_image
#from k_imageGen_stacked import ImageHelper, read_image
from tensorflow.keras.backend import manual_variable_initialization
#from newoptimizer import AdaBoundOptimizer
import time
from keras import backend as K
#from k_Unet_ASPP_2 import get_model as asppmodel
from functools import partial

def dice_coef(y_true, y_pred, epsilon=1e-6):
    
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(K.abs(y_pred * y_true), axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    return K.mean((numerator + epsilon) / (denominator + epsilon))
    
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
 
   
def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)
def weighted_categorical_crossentropy(weights):
    weights = tf.keras.backend.variable(weights)
    def loss(y_true, y_pred):

        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)
        return loss
        #print("Done.")
    return loss
'''
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T
'''

def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coef(y_true[:,:,:,:,label_index], y_pred[:,:, :,:,label_index])
    
def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f  

def get_model(hp,tile_size=64,modelno=0,isaux=False,dummy=False,inputlayer=None,previousoutputlayer = None):

    #tile_size = 64
    kernel_size = 3 
    #in_channels = 4
    out_channels = 4 
    # dropout_rate = 0.3
    up_sampling_stride = 2 
    # layers = 2 
    layers = hp.get('layers')
    # print("Inside fn - first  : ", layers)
    
    # features_root = 16
    features_root = hp.get('features_root')
    # learning_rate = 0.00001
    learning_rate = hp.get('learning_rate')
    is_simple = hp.get('is_simple')
    aspp_add_layers = OrderedDict()
    aspp_block = OrderedDict()

    # if(features_root>=16):
    #     hp.Choice('layers', [1, 2])
    #     layers = hp.get('layers')
    #     print("Inside fn - 16  : ", layers)
    
    # if(features_root>128):
    #     hp.Choice('layers', [2,3])
    #     layers = hp.get('layers')
    #     print("Inside fn - 128  : ", layers)
     
    input_shape = [tile_size, tile_size, tile_size,1]
  
    inputs = Input(shape=input_shape)
    inputsc = inputs
    if ((not inputlayer is None) and (not previousoutputlayer is None)):
        inputs = inputlayer
        inputsc = tf.keras.layers.concatenate([tf.keras.layers.Lambda(lambda x:x)(inputlayer),previousoutputlayer],axis = -1,name =f'concatenate_for_{modelno}')


    pool_size = hp.get('pool_size')
    regstr = hp.get('regstrength')

    # reduce features to 3 for VGG
    # in_node = Conv2D(filters = 3, kernel_size = 1, padding = "SAME", activation = "elu",
    #                 kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
    #                 name=f'Input_1x1_conv')(inputs)
    in_node = inputsc
    # in_node = inputs

    concatlist=[]
    concatlistnames=[]


    for i in range(0,layers*2+2):
        concatlist.append([])
        concatlistnames.append([])
    concatlist[1].append(in_node)
    concatlistnames[1].append('input')

    features = features_root
    #  (1)  Convolution/Downsampling: input - images, output - maxpooled image
    for layer in range(1,layers+1):
        
        features = features_root*layer
        

        #Simple Convolutions : n * n
        if(len(concatlist[layer])==1):
            inputforlayer = concatlist[layer][0]
        else :
            inputforlayer = concatenate(concatlist[layer],axis=-1,name=f'merge_for_{layer}_'+str(modelno))
        
        inputforlayer = Conv3D(filters=features,kernel_size=1,activation = "linear",name=f'block{layer}_sel'+"_"+str(modelno))(inputforlayer)

        features = features_root
            
        conv1 = Conv3D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu", 
                        kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
                        name=f'block{layer}_conv1'+"_"+str(modelno))(inputforlayer)

        

        conv2 = Conv3D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu", 
                        kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
                        name=f'block{layer}_conv2'+"_"+str(modelno))(conv1)
        
        
     
        if layer>2:
            conv2 = Conv3D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu",
                            kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
                            name=f'block{layer}_conv3'+"_"+str(modelno))(conv2)
        
        if False and layer == 2:
            conv2 = tf.keras.layers.BatchNormalization(name=f'block{layer}_conv1bn'+"_"+str(modelno))(conv2)
        
        else:
            
            for power in range(1-layer,layers-layer+2):
                if power<0:

                    toconcat = UpSampling3D(size = 2**abs(power),name = f'block{layer}_{power}_upsample'+"_"+str(modelno))(conv2)
                    concatlist[layers*2+2-layer+abs(power)].append(toconcat)
                    concatlistnames[layers*2+2-layer+abs(power)].append(f'block{layer}_{power}_upsample'+"_"+str(modelno))

                if power==0:

                    toconcat = conv2
                    concatlist[layers*2+2-layer].append(toconcat)
                    concatlistnames[layers*2+2-layer].append(f'block{layer}_{power}_direct_'+str(modelno))

                if power > 0:

                    toconcat = MaxPooling3D(pool_size=2**power,padding='SAME',name = f'block{layer}_{power}_pool'+"_"+str(modelno))(conv2)
                    if power!=layers-layer+1:
                        concatlist[layers*2+2-layer-abs(power)].append(toconcat)
                        concatlistnames[layers*2+2-layer-abs(power)].append(f'block{layer}_{power}_pool'+"_"+str(modelno))
                    concatlist[layer+power].append(toconcat)
                    concatlistnames[layer+power].append(f'block{layer}_{power}_pool'+"_"+str(modelno))

                    
                
            
    
    #bottleneck
    if len(concatlist[layers+1])==1:
        inputforbottle = concatlist[layers+1][0]
    else:
        inputforbottle = concatenate(concatlist[layers+1],axis=-1,name =f'inputmergeforbottle_'+str(modelno))
    
    inputforbottle= Conv3D(filters=features,kernel_size=1,activation = "linear",name=f'block{layers+1}_sel'+"_"+str(modelno))(inputforbottle)
    
    features = features_root*(layers+1)

    conv1 = Conv3D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu",
                    kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
                    name=f'block{layers+1}_conv1'+"_"+str(modelno))(inputforbottle)

    features = features_root 

    #conv1 = tf.keras.layers.BatchNormalization(name=f'block{layers+1}_conv1bn'+"_"+str(modelno))(tf.keras.layers.Activation('elu')(conv1))

    conv2 = Conv3D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu",
                    kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
                    name=f'block{layers+1}_conv2'+"_"+str(modelno))(conv1)
    
    if (layers + 1 )> 2:
        conv2 = Conv3D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu",
                        kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
                        name=f'block{layers+1}_conv3'+"_"+str(modelno))(conv2)
    
   

        
        

    else:

        for power in range (1,layers+1):

            toconcat = UpSampling3D(size = 2**abs(power),name = f'block{layers+1}_{power}_upsample'+"_"+str(modelno))(conv2)

            toconcat = Conv3D(filters = features,kernel_size = 2, padding ="SAME", activation ="elu", kernel_initializer = 'he_normal',
                              kernel_regularizer = regularizers.l2(regstr),name =f'up_block{layers+1}_{power}_upsampleconv'+str(modelno))(toconcat)

            concatlist[layers+1+power].append(toconcat)

            concatlistnames[layers+1+power].append(f'up_block{layers+1}_{power}_upsampleconv'+str(modelno))
    
    #deconvolution layers
    for layer in range(layers+2,layers*2+2):
        
        features = features_root*(layer)
        
        
        inputforlayer = concatenate(concatlist[layer],axis=-1,name=f'inputfor_{layer}_'+str(modelno))

        inputforlayer = Conv3D(filters=features,kernel_size=1,activation = "linear",name=f'block{layer}_sel'+"_"+str(modelno))(inputforlayer)

        features = features_root

        conv1 = Conv3D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu",
                        kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
                        name = f'up_block{layer}_conv1'+"_"+str(modelno))(inputforlayer)

        

        conv2 = Conv3D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu",
                        kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
                        name = f'up_block{layer}_conv2'+"_"+str(modelno))(conv1)

        if layer < layers*2:
            conv2 = Conv3D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu",
                            kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
                            name = f'up_block{layer}_conv3'+"_"+str(modelno))(conv2)
        if False and layer==layers*2:
            conv2 = tf.keras.layers.BatchNormalization(name=f'up_block{layer}_conv1bn'+"_"+str(modelno))(conv2)

        

            #Instead of MaxPooling - used Convolution with stride 2
            for power in range(1,layers*2-layer+2):

                toconcat = UpSampling3D(size = 2**abs(power),name = f'block{layer}_{power}_upsample'+"_"+str(modelno))(in_node)

                toconcat = Conv3D(filters = features,kernel_size = 2, padding ="SAME", activation ="elu", kernel_initializer = 'he_normal',
                                  kernel_regularizer = regularizers.l2(regstr),name =f'up_block{layer}_{power}_upsampleconv'+str(modelno))(toconcat)

                concatlist[layer+power].append(toconcat)

                concatlistnames[layer+power].append(f'up_block{layer}_{power}_upsampleconv'+str(modelno))

            


        else:

            for power in range(1,layers*2-layer+2):
                toconcat = UpSampling3D(size = 2**abs(power),name = f'block{layer}_{power}_upsample'+"_"+str(modelno))(conv2)
                toconcat = Conv3D(filters = features,kernel_size = 2, padding ="SAME", activation ="elu", kernel_initializer = 'he_normal',
                                  kernel_regularizer = regularizers.l2(regstr),name =f'up_block{layer}_{power}_upsampleconv'+str(modelno))(toconcat)
                concatlist[layer+power].append(toconcat)
                concatlistnames[layer+power].append(f'up_block{layer}_{power}_upsampleconv'+str(modelno))

            if layer == layers*2+1:
                 in_node = conv2
    # for i in range(1,5):

    #     aspp_conv1 = Conv2D(filters = features , kernel_size = kernel_size, padding = "SAME", activation = "elu", 
    #                             kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
    #                             dilation_rate = 6*i, name=f'aspp_dil_stem{i}_dr{6*i}'+"_"+str(modelno))(in_node)
    #     aspp_conv2 = Conv2D(filters = features , kernel_size = 1, padding = "SAME", activation = "elu",
    #                             kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
    #                             name=f'aspp_1x1xF_stem{i}_dr{6*i}_1'+"_"+str(modelno))(aspp_conv1)
    #     aspp_conv3 = Conv2D(filters = out_channels , kernel_size = 1, padding = "SAME", activation = "linear",
    #                             kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(regstr),
    #                             name=f'aspp_1x1xF_stem{i}_dr{6*i}_2'+"_"+str(modelno))(aspp_conv2)
    #     aspp_block[i] = aspp_conv3

    # #summation
    # in_node = tf.keras.layers.Add()(list(aspp_add_layers.values()) + list(aspp_block.values()))

    #print(concatlistnames)
    in_node = Conv3D(filters = out_channels, kernel_size = 1, padding = "SAME", activation = "linear",
                    kernel_initializer = 'he_normal', name = f'logits_{out_channels}_out')(in_node)


    logits = tf.keras.layers.Activation('softmax')(in_node)
    # logits = Conv2D(filters = out_channels, kernel_size = 1, padding = "SAME", activation = "softmax",
    #                 kernel_initializer = 'he_normal', name = f'logits_{out_channels}_out')(in_node)
    # logits = UpSampling2D(size = pool_size, name=f'Output_Up_Sample_by{pool_size}')(logits)
    
    label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(out_channels)]
    
    if isaux:
        model = Model(inputs = inputs, outputs = in_node)
        model.compile(optimizer = SGD(lr = hp.get('learning_rate'),momentum = 0.9,nesterov = True), loss = "categorical_crossentropy", metrics = ["accuracy", dice_coef, label_wise_dice_metrics])
        model.summary(line_length=150)
        #optimizer = SGD(lr = hype.get('learning_rate'),momentum=0.9,nesterov=True)
        #focal_tversky_loss
        return model
    else :
        model = Model(inputs = inputs, outputs = logits)
        model.compile(optimizer =SGD(lr = hp.get('learning_rate'),momentum = 0.9,nesterov = True), loss = "categorical_crossentropy" , metrics = ["accuracy", dice_coef, label_wise_dice_metrics])
        model.summary(line_length=150)
        #optimizer = SGD(lr = hype.get('learning_rate'),momentum=0.9,nesterov=True)
        return model
            
           



