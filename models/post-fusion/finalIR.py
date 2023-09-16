# -*- coding: utf-8 -*-
"""
Created on Wed May 17 23:25:39 2023

@author: AL-NABAA
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys, os
sys.path.append("../../dataset/wmca")
print((os.path.realpath(__file__)).__str__())
import h5py
import numpy as np
import cv2
import tensorflow as tf
from keras.layers import Dense, Dropout, activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn import metrics
from wmca import WMCA 
import gc 
import matplotlib.pyplot as plt

loaddb = WMCA('../../dataset')

modalities = ['I']
for mod in modalities:    
    trainimages = None
    trainlabels = None
    testimages = None
    testlabels = None
    collected = gc.collect()
    print ("collected", collected)       
    trainimages, trainlabels, testimages, testlabels = loaddb.readDataset(mod,0.70)
    #######################################################
    # define the model
    print("start building the model")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='tanh', input_shape=(128, 128, len(mod))))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(100, kernel_size=(3, 3), activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(140, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    #######################################################
    # compile the model
    print("start compiling the model")
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    #######################################################
    # fit the model on the training data
    print("start training the model")
    model.fit(trainimages, trainlabels, validation_data=(testimages, testlabels) , epochs=6, batch_size=200)
    pred = model.predict(testimages)
    pred = pred[:,0]
    pred = list(map(lambda x: 0 if x<0.5 else 1, pred))
    matrix = metrics.confusion_matrix(testlabels, pred)
    tn, fp, fn, tp = matrix.ravel()
    #######################################################
    #curve                 
    fpr,tpr, thresholds = metrics.roc_curve(testlabels, pred)
    roc_auc = metrics.auc (fpr, tpr)
    print("AUC=", roc_auc, mod)
    print (thresholds)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc , estimator_name=mod)
    display.plot()
    plt.show()
    ########################################################
    #print of result                  
    print("--------------------------------")
    print("       ", mod)
    print("--------------------------------")
    print (matrix)
    print("TN",tn,"FP",fp,"FN",fn,"TP",tp)
    apcer = fp/(tn+fp)*100
    bpcer = fn/(fn+tp)*100
    print ("APCER", apcer)
    print("BPCER", bpcer)
    print ("ACER",(apcer + bpcer)/2)
    print()
    #######################################################
    #save of result               
    model.save('model_I.h5')
    print('Model I has been saved.')
