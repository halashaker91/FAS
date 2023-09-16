# -*- coding: utf-8 -*-
"""
Created on Sun May 28 15:55:36 2023

@author: AL-NABAA
"""
import sys, os
sys.path.append("../../dataset/wmca")
print((os.path.realpath(__file__)).__str__())
import csv
import h5py
import numpy as np
import cv2
import tensorflow as tf
from keras.layers import Dense, Dropout, activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn import metrics
from wmca import WMCA 
import gc 
from fusion import Fusion

loaddb = WMCA('../../dataset')

# for mod in modalities:    
trainimagesC, trainlabelsC, testimagesC, testlabelsC = loaddb.readDataset('C',0.70)
trainimagesD, trainlabelsD, testimagesD, testlabelsD = loaddb.readDataset('D',0.70)
trainimagesI, trainlabelsI, testimagesI, testlabelsI = loaddb.readDataset('I',0.70)
trainimagesT, trainlabelsT, testimagesT, testlabelsT = loaddb.readDataset('T',0.70)

modelC= tf.keras.models.load_model('model_C.h5')
modelD= tf.keras.models.load_model('model_D.h5')
modelI= tf.keras.models.load_model('model_I.h5')
modelT= tf.keras.models.load_model('model_T.h5')

predC= modelC.predict(trainimagesC)
predD= modelD.predict(trainimagesD)
predI= modelI.predict(trainimagesI)
predT= modelT.predict(trainimagesT)

header_row = 'C , D, I, T, label\n'
f = open ('fusiondata.csv', 'w')
f.write(header_row)
for i in range(0, len(predC)):
    classC = predC[i][0]
    classD = predD[i][0]
    classI = predI[i][0]
    classT = predT[i][0]
    f.write(str(classC)+','+str(classD)+','+str(classI)+','+str(classT)+','+str(trainlabelsC[i])+'\n')
f.close()    
    
    
  