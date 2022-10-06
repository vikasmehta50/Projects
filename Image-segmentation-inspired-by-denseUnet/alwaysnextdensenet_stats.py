import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
from collections import OrderedDict
from numpy import linalg as LA
import math
import csv
import copy
import tensorflow
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
#from k_imageGen8 import getImage, one_hot_labels
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)

def printweightsummary(newmodel,epochno,modelno,acc,suffix):
    printlist = ['lno.','layername','alive','dead','ratio a/d','minimumnorm','10thperc','25thperc','50thperc','90thperc','99thperc']
    layerno =[]
    layernorms = []
    for idx, layer in enumerate(newmodel.layers):  

        if(layer.trainable == False):
            continue                                      
    
        if(layer.__class__.__name__!='Conv3D' and layer.__class__.__name__!='Conv3DTranspose'):
            continue
        weights=np.array(layer.get_weights()[0])
        

        
        layerno.append(idx)
        #print("shape of weights is:")
        #print(weights.shape)
        newlist=[]
        for j in range(0,weights.shape[3]): #the shape needs to be evaluated
            newlist.append(np.abs(weights[:,:,:,j]))

        layernorms.append(newlist)
    cnt =0 
    modelist =[]
    for lay in layernorms:

        msk=np.array(lay)
        msk = msk.flatten()
        greatprint = msk[abs(msk)>0.001]
        lessprint = msk[abs(msk)<=0.001]
        try:
          csvlist =[layerno[cnt],newmodel.layers[layerno[cnt]].name,str(len(greatprint)),len(lessprint),len(greatprint)/(len(lessprint)+len(greatprint)),round(np.min(lay),6),round(np.percentile(lay,10),6),round(np.percentile(lay,25),6),round(np.percentile(lay,50),6),round(np.percentile(lay,90),6),round(np.percentile(lay,99),6)]
          modelist.append(csvlist)
        except:
          print("Exception occured in stats")
          print(len(greatprint))
          print(len(lessprint))
        cnt+=1
    with open(f'/content/drive/MyDrive/newibsr/IBSR/IBSR_nifti_stripped/stacked_stats/statsformodelblock_{modelno}_dense'+suffix+'.csv','a',newline='') as csvfile:

        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([str(epochno),str(acc)])
        csvwriter.writerow(printlist)
        csvwriter.writerows(modelist)