# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:43:25 2023
@author: AL-NABAA
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
from fusion import Fusion
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
loaddb = WMCA('../../dataset')
   
trainimagesC, trainlabelsC, testimagesC, testlabelsC = loaddb.readDataset('C',0.70)
trainimagesD, trainlabelsD, testimagesD, testlabelsD = loaddb.readDataset('D',0.70)
trainimagesI, trainlabelsI, testimagesI, testlabelsI = loaddb.readDataset('I',0.70)
trainimagesT, trainlabelsT, testimagesT, testlabelsT = loaddb.readDataset('T',0.70)

modelC= tf.keras.models.load_model('model_C.h5')
modelD= tf.keras.models.load_model('model_D.h5')
modelI= tf.keras.models.load_model('model_I.h5')
modelT= tf.keras.models.load_model('model_T.h5')

predC= modelC.predict(testimagesC)
predD= modelD.predict(testimagesD)
predI= modelI.predict(testimagesI)
predT= modelT.predict(testimagesT)

fusion = Fusion()
tpm = tpw = tpavg = tpstc = 0
tnm = tnw = tnavg = tnstc = 0
fpm = fpw = fpavg = fpstc = 0
fnm = fnw = fnavg = fnstc = 0
predictions = [0,0,0,0]
classes  = [0,0,0,0]
prob = [0,0,0,0]
for i in range(0, len(predC)):
    predictions[0] = predC[i][0]
    predictions[1] = predD[i][0]
    predictions[2] = predI[i][0]
    predictions[3] = predT[i][0]      
    for j in range (0,4):
        classes[j] = round(predictions[j])
        if classes[j]==1:
            prob[j] = predictions[j]
        else:
            prob[j] = 1 - predictions[j]    
    print("Predictions", predictions)
    print("Classes", classes)
    print("Probabilites", prob)       
    label = testlabelsC[i]    
    finalpred_majority = fusion.majority_voting (classes)
    finalpred_weighted = fusion.weighted_voting(classes, prob) 
    finalpred_average  = fusion.average_pooling(predictions)
    finalpred_stacking_classifier = fusion.stacking_classifier([predictions])
    print ('stacking_classifier', finalpred_stacking_classifier)   
    if label==1:
        if finalpred_majority == 1:
            tpm += 1
        else:
            fpm += 1
    else:
        if finalpred_majority == 1:
            fnm += 1
        else:
            tnm += 1
##################################################################
    if label==1:
        if finalpred_weighted == 1:
            tpw += 1
        else:
            fpw += 1
    else:
        if finalpred_weighted == 1:
            fnw += 1
        else:
            tnw += 1  
##################################################################            
    if label==1:
        if finalpred_average == 1:
            tpavg += 1
        else:
            fpavg += 1
    else:
        if finalpred_average == 1:
            fnavg += 1
        else:
            tnavg += 1    
#################################################################   
    if label==1:
        if finalpred_stacking_classifier == 1:
            tpstc += 1
        else:
            fpstc += 1
    else:
        if finalpred_stacking_classifier == 1:
            fnstc += 1
        else:
            tnstc += 1  
##################################################################
print("TN",tnm,"FP",fpm,"FN",fnm,"TP",tpm)
apcer = fpm/(tnm+fpm)*100
bpcer = fnm/(fnm+tpm)*100
print ("APCER-m", apcer)
print("BPCER-m", bpcer)
print ("ACER-m",(apcer + bpcer)/2)
##################################################################
print("TN",tnw,"FP",fpw,"FN",fnw,"TP",tpw)
apcer = fpw/(tnw+fpw)*100
bpcer = fnw/(fnw+tpw)*100
print ("APCER-w", apcer)
print("BPCER-w", bpcer)
print ("ACER-w",(apcer + bpcer)/2)
##################################################################
print("TN",tnavg,"FP",fpavg,"FN",fnavg,"TP",tpavg)
apcer = fpavg/(tnavg+fpavg)*100
bpcer = fnavg/(fnavg+tpavg)*100
print ("APCER-avg", apcer)
print("BPCER-avg", bpcer)
print ("ACER-avg",(apcer + bpcer)/2)
##################################################################
print("TN",tnstc,"FP",fpstc,"FN",fnstc,"TP",tpstc)
apcer = fpstc/(tnstc+fpstc)*100
bpcer = fnstc/(fnstc+tpstc)*100
print ("APCER-stc", apcer)
print("BPCER-stc", bpcer)
print ("ACER-stc",(apcer + bpcer)/2)
print()
##################################################################
    