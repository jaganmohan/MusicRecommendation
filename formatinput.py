#!/usr/bin/python

import numpy as np
import os
import pickle
#import cPickle as pickle
import sys


path = 'melspec'


filenames = os.listdir(path)
data = {}
i = 0
for file in filenames:
    file_path = os.path.join(path, file)
    #print (file_path)
    with open(file_path, 'rb') as f:
        try:
            soundId = os.path.splitext(file)[0]
            #print (soundId)
            content = f.read()
            pp = pickle.loads(content,encoding='latin1')

            ##pp= pickle.load(f)
            pp = np.asarray(pp)   
            #print (pp[0])
            data[str(i)] = pp
            i+=1
        except Exception as e:
            print ("Error occurred" + str(e))
        
mask = np.genfromtxt('mask.csv',delimiter=',')
mask = mask.astype(int)
total = mask.shape[0]
X = 599
Y = 128
Z = 2
labels = np.genfromtxt('V.csv',delimiter=',')
userfactors = np.genfromtxt('U.csv',delimiter = ',')
matrix = np.genfromtxt('cmatrix.csv',delimiter=',')
matrix = matrix[:,:900]
tdata = np.ndarray((total, X, Y,Z), dtype=np.uint8)
tlabel = np.ndarray((total,labels.shape[1]))
for j,k in enumerate(mask[:900]):
    tdata[j] = data[str(k)]
    tlabel[j] = labels[j]

   

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Reshape, Flatten
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import losses,optimizers,utils
from keras.optimizers import Adam, SGD
N = labels.shape[1]

#def Loss_latentvecs(true, pred):
    
def predictedprob(pred):
    return np.dot(userfactors,pred)
def featurevectorloss(true,pred):
    predprob = predictedprob(np.transpose(pred))
    actprob = np.zeros_like(matrix)
    actprob[matrix>0]=1
    loss = np.sum((actprob-predprob)**2)
    return loss

def MusicModel(input_shape):
    dropout = 0.2
    inp = Input(shape = input_shape, name = 'input1')
    x = Conv2D(32,(4,4),padding = 'same')(inp)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D(4)(x)
    x = Conv2D(64,(4,4),padding = 'same')(x)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)  
    x = Conv2D(128,(4,4),padding = 'same')(x)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256,(4,4),padding = 'same')(x)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    print (K.shape(x))
    x = Dense(1024,activation='relu')(x)
    x = Dense(N)(x)
    model = Model(inp, x)
    return model 

model = MusicModel(input_shape=(599,128,2))    
model.summary()
opt =  SGD(lr=0.0001)
model.compile(optimizer=opt, loss=featurevectorloss)
checkpoint = ModelCheckpoint('weight.h5', monitor='val_loss',save_best_only=True)
model.save('my_model.h5')
    
model.fit(tdata,tlabel,batch_size=900, epochs=80, verbose=1, shuffle=True,callbacks=[checkpoint])
