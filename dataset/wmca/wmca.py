# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:46:31 2023
@author: lenovo
"""
import os
import cv2
import numpy as np
import h5py
from sklearn.preprocessing import normalize 

class WMCA:
    
    def __init__(self, path):
        self.path = path
      # self.attacktxt = os.path.join(path, 'attack_illustration_files.csv')
      # self.bonafidetxt = os.path.join(path, 'test_private_list.txt')
        self.attacktxt = 'attack_illustration_files.csv'
        self.bonafidetxt = 'bonafide_illustration_files.csv'        
        self.attdsfiles = []
        self.bondsfiles = []
        
    def readTXTFiles(self, modality='CDIT'):
        mod = 0
        if modality=='RGB':
            modpath = '/WMCA_preprocessed_RGB/'
        else:
            modpath = '/WMCA_preprocessed_CDIT/'
        attackfilename =  self.path+ modpath + self.attacktxt
        bonafidefilename =  self.path+ modpath + self.bonafidetxt
        attackfilenames = []
        attacklabels = []
        with open(attackfilename) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                s = line.split()
                attackfilenames.append(self.path+modpath+s[mod])
        bonafidefilenames = []
        with open(bonafidefilename) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                s = line.split()
                bonafidefilenames.append(self.path+ modpath +s[mod])
        return attackfilenames, bonafidefilenames
    
    def getImage(self, arr, modality):  
      # print ("mod", modality)
        arr = arr.transpose(1,2,0)
        if modality=="RGB":
            image = cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)
        else:            
            image = np.zeros((arr.shape[0],arr.shape[1],0))
            if modality.find('C')!=-1 :
                c_img = arr[:,:,0]
                image = np.insert(image, 0, c_img, axis=2)     
            if modality.find('D')!=-1 :
                d_img = arr[:,:,1]
                image = np.insert(image, 0, d_img, axis=2) 
            if modality.find('I')!=-1 :
                i_img = arr[:,:,2]
                image = np.insert(image, 0, i_img, axis=2) 
            if modality.find('T')!=-1 :
                t_img = arr[:,:,3]
                image = np.insert(image, 0, t_img, axis=2)      
      # print(image.shape)
        return image
        
    def readDataset(self, modality='CDIT', trainrate=0.7):
        images = []
        attdstext, bondstext = self.readTXTFiles(modality)
        print("length", len(attdstext))
        atttrainnum = int(len(attdstext) * trainrate)
      # testnum = len(dstext) - trainnum
        attdstext = np.array(attdstext)
        atttrainfilenames = attdstext[0:atttrainnum]
        atttestfilenames = attdstext[atttrainnum:len(attdstext)]
        atttrainlabels = np.zeros((atttrainnum))
        atttestlabels = np.zeros((len(attdstext)-atttrainnum))
        print("length", len(bondstext))
        bontrainnum = int(len(bondstext) * trainrate)
      # testnum = len(dstext) - trainnum
        bondstext = np.array(bondstext)
        bontrainfilenames = bondstext[0:bontrainnum]
        bontestfilenames = bondstext[bontrainnum:len(bondstext)]
        bontrainlabels = bondstext[0:bontrainnum]
        bontestlabels = bondstext[bontrainnum:len(bondstext)]
        print("SIZES = ",len(atttrainfilenames), len(atttestfilenames))
        print("SIZES = ",len(bontrainfilenames), len(bontestfilenames))
        i = 0        
        trainimages = []
        trainlabels = []
        testimages = []
        testlabels = []
        print ("att train")
        for filename in atttrainfilenames:
          # print(filename)
            if filename != '.':
                with h5py.File(filename, "r") as f:
                     for k in f.keys():
                        f1 = f.get(k)
                      # print ("f1", f1)
                        g = f1.get('array')
                      # print ("g", g)
                        g = np.array(g)
                      # print ("g", g)
                        image = self.getImage(g,modality)
                        trainimages.append(image)
                        trainlabels.append(0)
        print("bon train") 
        for filename in bontrainfilenames:
            if filename != '.' and filename !='/':
                with h5py.File(filename, "r") as f:
                    for k in f.keys():
                        f1 = f.get(k)
                        g = f1.get('array')
                        g = np.array(g)
                        image = self.getImage(g,modality)
                        trainimages.append(image)
                        trainlabels.append(1)
        print("att test")
        for filename in atttestfilenames:
           #print(filename)
            if filename != '.':
                with h5py.File(filename, "r") as f:
                    for k in f.keys():
                        f1 = f.get(k)
                        g = f1.get('array')
                        g = np.array(g)
                        image = self.getImage(g,modality)
                      # print (image.min(),image.max())
                        testimages.append(image)
                        testlabels.append(0)
        print("bon test")
        for filename in bontestfilenames:
            if filename != '.':
                with h5py.File(filename, "r") as f:
                    for k in f.keys():
                      # print("key", k)
                        f1 = f.get(k)
                        g = f1.get('array')
                        g = np.array(g)
                        image = self.getImage(g,modality)
                        # print (image.min(),image.max())
                        testimages.append(image)
                        testlabels.append(1)            
        trainimages = np.array (trainimages)
        trainlabels = np.array (trainlabels)
        testimages = np.array (testimages)
        testlabels = np.array (testlabels)                                   
        return trainimages, trainlabels, testimages, testlabels   
########################################################################    
#cross-validation     
    def createDatasetKFold(self, modality='CDIT'):         
        attdstext, bondstext = self.readTXTFiles(modality)
        print("att", len(attdstext))
        print("bon", len(bondstext))
        self.attdsfiles = np.array(attdstext)
        self.bondsfiles = np.array(bondstext)
    def readDatasetKFold(self, modality="CDIT", f=1, k=5):
        attlen = len(self.attdsfiles)
        a = (f-1)*int(attlen/k) 
        b = a + int(attlen/k - 1)
        print("attlen", attlen, a, b)
        atttestfiles = self.attdsfiles[a:b]       
        if a==0:
            t1 = []
        else:    
            t1 = self.attdsfiles[0:a]
        t2 = self.attdsfiles[b+1:]
        print("att test", len(atttestfiles), "atttrain", len(t1),len(t2))
        atttrainfiles = np.concatenate((t1,t2))       
##################################################        
        
        bonlen = len(self.bondsfiles)
        print("bonlen", bonlen)
        a = (f-1)*int(bonlen/k) 
        b = a + int(bonlen/k - 1)
        bontestfiles = self.bondsfiles[a:b]
        if a==0:
            t1 = []
        else:    
            t1 = self.bondsfiles[0:a-1]
        t2 = self.bondsfiles[b+1:]
        bontrainfiles = np.concatenate((t1,t2))        
###################################################        
        trainimages = []
        trainlabels = []
        testimages = []
        testlabels = []
        print ("att train")
        for filename in atttrainfiles:
          # print(filename)
            if filename != '.':
                with h5py.File(filename, "r") as f:
                     for k in f.keys():
                        f1 = f.get(k)
                        g = f1.get('array')
                        g = np.array(g)
                        image = self.getImage(g,modality)
                        trainimages.append(image)
                        trainlabels.append(0)
        print("bon train") 
        for filename in bontrainfiles:
            if filename != '.' and filename !='/':
                with h5py.File(filename, "r") as f:
                    for k in f.keys():
                        f1 = f.get(k)
                        g = f1.get('array')
                        g = np.array(g)
                        image = self.getImage(g,modality)
                        trainimages.append(image)
                        trainlabels.append(1)
        print("att test")
        for filename in atttestfiles:
           #print(filename)
            if filename != '.':
                with h5py.File(filename, "r") as f:
                    for k in f.keys():
                        f1 = f.get(k)
                        g = f1.get('array')
                        g = np.array(g)
                        image = self.getImage(g,modality)
                      # print (image.min(),image.max())
                        testimages.append(image)
                        testlabels.append(0)
        print("bon test")
        for filename in bontestfiles:
            if filename != '.':
                with h5py.File(filename, "r") as f:
                    for k in f.keys():
                        # print("key", k)
                        f1 = f.get(k)
                        g = f1.get('array')
                        g = np.array(g)
                        image = self.getImage(g,modality)
                      # print (image.min(),image.max())
                        testimages.append(image)
                        testlabels.append(1) 
        trainimages = np.array (trainimages)
        trainlabels = np.array (trainlabels)
        testimages = np.array (testimages)
        testlabels = np.array (testlabels)                   
        return trainimages, testimages, trainlabels, testlabels    
    ###################################################
    
  