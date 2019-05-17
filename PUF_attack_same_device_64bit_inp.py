# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:49:21 2019

@author: Shiva PC
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
import seaborn as sns

np.random.seed(1257)
#tf.set_random_seed(1257)

def readData():
    data = pd.ExcelFile('data.xlsx')
    df = data.parse('Sheet1')
    df = df.to_numpy()
    df1 = np.where(df==0,df-1,df+0)
    
    dat = np.zeros((181,65,24),dtype = np.int64)
    bin = np.zeros((181,65,24),dtype = np.int64)
    
    volt = np.arange(-90,91)
    for i in range(np.size(df1,0)):
        dat[int(math.floor(i/24)),:,i%24] = np.append(volt[int(math.floor(i/24))],df1[i,:])
        bin[int(math.floor(i/24)),:,i%24] = np.append(volt[int(math.floor(i/24))],df[i,:])
    return dat,bin

def Ploting_Resp(i,data):
    plt.imshow(dat[:,:,i],cmap='gray')

def Shuffle_dat(dat):
    for i in range(np.size(dat,2)):
        shuf = dat[:,:,i]
        np.random.shuffle(shuf)
        dat[:,:,i] = shuf
        return dat
    
def dataPart(data,ratio):
    val = int(ratio*np.size(data,0))
    train = data[:(val),:,:]
    test = data[val:(np.size(data,0)),:,:]
    return train,test

def func(x):
    a[x < 0] = 0
    a[x >= 0] = 1
    return a

def num_test(train,test):
    y = func(-1,1,1)
    print(y)
    
def logistReg(x_t,y_t,test_x, test_y):
    model = Sequential()
    model.add(Dense(60,activation = 'sigmoid',kernel_regularizer = L1L2(l1=0,l2=0.1),input_dim = 65))
    model.add(Dense(64,activation = 'sigmoid',use_bias = False,bias_initializer = 'zeros', kernel_regularizer = L1L2(l1=0,l2=0.1),input_dim = 60))    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
    model.fit(x_t,y_t, epochs = 50, validation_data = (test_x,test_y))
    score = model.evaluate(test_x,test_y)
    lo = model.predict(test_x)
    print('Test Accuracy', score[1]*100)
    print(model.summary())
    return lo

def CorPlot(lo,test_y):
    corr = np.zeros((np.size(lo,0),np.size(lo,0)))
        
    for i in range(np.size(lo,0)):
        for j in range(np.size(lo,0)):
            cor = np.corrcoef(lo[i,:],test_y[j,:])
            corr[i,j] = cor[0,1]
    add = np.isnan(corr)
    corr[add] = 0
    ax = sns.heatmap(corr,vmin = 0, vmax =1, linewidths = 0.5, cmap = 'cool')
    #ax.title('Correlation')
    return corr

def binConv(trainx):
    sin = np.sign(trainx)
    val = np.abs(trainx)
    out = np.zeros((np.size(trainx,0),65))
    for i in range(np.size(trainx,0)):
        outp = np.binary_repr(val[i],64)
        for j in range(65):
            if j<64:
                out[i,j] = int(outp[j])
            if j==64:
                out[i,j] = sin[i]
    ou = np.where(out==0,out-1,out+0)
    return ou
    
start = time.time()
dat,bin = readData()
end = time.time()
print('Time taken to load dataset ',end-start,'seconds')

dat = Shuffle_dat(dat)    
bin = Shuffle_dat(bin)
ratio = 0.80
train,test = dataPart(dat,ratio)
bin_train, bin_test = dataPart(bin,ratio)

num_Test = False
log_reg = True

train_x = train[:,0,0]
train_y = bin_train[:,1:65,0]
test_x = test[:,0,0]
test_y = bin_test[:,1:65,0]

train_x = binConv(train_x)
test_x = binConv(test_x)

if log_reg:
    lo = logistReg(train_x, train_y, test_x, test_y)
    lo = np.where(lo>=0.5,1,0)
    cor = CorPlot(lo,test_y)
    print('Maximum correlation',np.max(cor))
    print('Minimum correlation', np.min(cor))
    print('Average correlation', np.average(cor))
  
