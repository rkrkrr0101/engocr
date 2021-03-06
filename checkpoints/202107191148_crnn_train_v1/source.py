# In[0]


# In[1]
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation,BatchNormalization,Lambda,Input,Conv2D,Bidirectional,LSTM, Concatenate, MaxPooling2D,GlobalAveragePooling2D
import tensorflow.keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
import numpy as np
from generater import crnnGenerator
import logging
logger=tf.get_logger()
logger.setLevel(logging.ERROR)

# In[2]
def convert_to_onehot(data):
    #Creates a dict, that maps to every char of alphabet an unique int based on position
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    char_to_int = dict((c,i) for i,c in enumerate(alphabet))
    
    encoded_data = []
    #Replaces every char in data with the mapped int
    encoded_data.append([char_to_int[char] for char in data])
    
    

    #This part now replaces the int by an one-hot array with size alphabet
    one_hot = []
    for value in encoded_data:
        #At first, the whole array is initialized with 0
        
        for indexvalue in value:
            letter = [0 for _ in range(len(alphabet))]
            #Only at the number of the int, 1 is written

            letter[indexvalue] = 1
            
            one_hot.append(letter)
    return one_hot

def convert_to_lable(data):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return list(map(lambda x: alphabet.index(x), data))



convert_to_lable('abcd453')

# In[3]
def crnn():
    x=image_input=Input(name='image_input',shape=(256,32,1))
    """
    minx=Input(name='minx',shape=[1],dtype='float32')
    miny=Input(name='miny',shape=[1],dtype='float32')
    maxx=Input(name='maxx',shape=[1],dtype='float32')
    maxy=Input(name='maxy',shape=[1],dtype='float32')
    def cropimage(img,minx=0,miny=0,maxx=0,maxy=0):
        
        return tf.image.crop_to_bounding_box(img,minx,miny,maxx-minx,maxy-miny)
    x=Lambda(cropimage)([x,minx,miny,maxx,maxy])
    
    """
        
    x=Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1_1')(x)#128 64 64
    x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1', padding='same')(x)#64 32 64
    
    x=Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2_1')(x)#64 32 128
    x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2', padding='same')(x)# 32 16 128

    x=Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3_1')(x)#32 16 256
    x=Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3_2')(x)#32 16 256
    x=MaxPooling2D(pool_size=(2, 2), strides=(1, 2), name='pool3', padding='same')(x)# 32 8 256
    
    x=Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4_1')(x)#32 8 512
    x=BatchNormalization(name='batch1')(x)
    
    x=Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5_1')(x)#32 8 512
    
    x=BatchNormalization(name='batch2')(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=(1, 2),padding='valid', name='pool5')(x)# 32 4 512
    
    x=aa=Conv2D(512, (2, 2), strides=(1, 1), activation='relu', padding='valid', name='conv6_1')(x)# 32 4 512
    s=x.shape
    x=keras.layers.Reshape((s[1],s[3]),name='reshape')(x)#32 2048
    x=Bidirectional(LSTM(256, return_sequences=True))(x)
    x=Bidirectional(LSTM(256, return_sequences=True))(x)#32 512
    
    num_classes=36+1
    x=keras.layers.Dense(num_classes, name='dense2')(x)#?????????+??????#1953 36
    
    x=y_pred= Activation('softmax', name='softmax')(x)#1953 36
    
   # model_pred=keras.models.Model(inputs=[image_input,minx,miny,maxx,maxy],outputs=x)
    model_pred=keras.models.Model(inputs=image_input,outputs=x)
    
    maxstringlen=s[1]
    #maxstringlen=100
    
    def ctc_lambda_func(args):
        y_pred,labels, input_length, label_length = args
          #y_pred = y_pred[:, 2:, :] 
        return K.ctc_batch_cost(y_true=labels,y_pred=y_pred,input_length=input_length,label_length=label_length)    
        #return tf.nn.ctc_loss(labels=labels,logits=y_pred,logit_length=input_length,label_length=label_length,logits_time_major=False)    
    
    labels = Input(name='labels', shape=[maxstringlen], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64') 
    
    ctcloss=Lambda(ctc_lambda_func,output_shape=(1,),name='ctcloss')([y_pred,labels,input_length,label_length])
    
   # model_train=keras.models.Model(inputs=[image_input,minx,miny,maxx,maxy,labels,input_length,label_length],outputs=ctc_loss)
    model_train=keras.models.Model(inputs=[image_input,labels,input_length,label_length],outputs=ctcloss)
    
    return model_train,model_pred,maxstringlen

    

# In[4]
traincrnn,predcrnn,inputlength=crnn()
traincrnn.summary()

# In[5]
inputlength

# In[6]
cropvalpath='D:/engocr/croptestimage/'
croptrainpath='D:/engocr/croptestvalimage/'
cropvaljson='D:/engocr/valcrnn.json'
batch=32
traingen=crnnGenerator(imgpath=croptrainpath,labelpath=cropvaljson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=30,inputlen=inputlength)
print('trainstart')
traingen.build_data()
print('valstart')
valgen=crnnGenerator(imgpath=cropvalpath,labelpath=cropvaljson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=30,inputlen=inputlength)
valgen.build_data()
print('valend')

# In[7]
traincrnn.compile(optimizer=tf.keras.optimizers.Adadelta() ,loss={'ctcloss':lambda y_true,y_pred:y_pred})

# In[8]
from tensorflow.python.keras.callbacks import TensorBoard  ## TensorBoard ??? import?????????.
modelver='crnn_train_v1'
checkdir = './checkpoints/' + datetime.datetime.now().strftime('%Y%m%d%H%M') + '_' + modelver
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())
log_dir = "./logs/fit/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard =TensorBoard(log_dir=log_dir,histogram_freq=1)

