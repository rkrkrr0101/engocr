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
import string
def convert_to_lable(data):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    alphabet87 = string.ascii_lowercase + string.ascii_uppercase + string.digits + ' +-*.,:!?%&$~/()[]<>"\'@#_'
    return list(map(lambda x: alphabet87.index(x), data))



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
    x=BatchNormalization(name='batchnorm1')(x)
    
    x=Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5_1')(x)#32 8 512
    
    x=BatchNormalization(name='batchnorm2')(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=(1, 2),padding='valid', name='pool5')(x)# 32 4 512
    
    x=aa=Conv2D(512, (2, 2), strides=(1, 1), activation='relu', padding='valid', name='conv6_1')(x)# 32 4 512
    s=x.shape
    x=keras.layers.Reshape((s[1],s[3]),name='reshape_1')(x)#32 2048
    x=Bidirectional(LSTM(256, return_sequences=True,name='bidirectional_1'))(x)
    x=Bidirectional(LSTM(256, return_sequences=True,name='bidirectional_1'))(x)#32 512
    
    num_classes=87
    x=keras.layers.Dense(num_classes, name='dense1')(x)#알파벳+숫자#1953 36
    
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
traincrnn.load_weights('weights.300000.h5',by_name=True)
freeze = ['conv1_1',
          'conv2_1',
          'conv3_1', 
          'conv3_2', 
          'conv4_1',
          'conv5_1',
          #'conv6_1',
          #'lstm1',
          #'lstm2'
         ]
for layer in traincrnn.layers:
    layer.trainable=not layer.name in freeze

traincrnn.summary()

# In[5]
inputlength

# In[6]
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
    x=BatchNormalization(name='batchnorm1')(x)
    
    x=Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5_1')(x)#32 8 512
    
    x=BatchNormalization(name='batchnorm2')(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=(1, 2),padding='valid', name='pool5')(x)# 32 4 512
    
    x=aa=Conv2D(512, (2, 2), strides=(1, 1), activation='relu', padding='valid', name='conv6_1')(x)# 32 4 512
    s=x.shape
    x=keras.layers.Reshape((s[1],s[3]),name='reshape_1')(x)#32 2048
    x=Bidirectional(LSTM(256, return_sequences=True,name='bidirectional_1'))(x)
    x=Bidirectional(LSTM(256, return_sequences=True,name='bidirectional_1'))(x)#32 512
    
    num_classes=87
    x=keras.layers.Dense(num_classes, name='dense1')(x)#알파벳+숫자#1953 36
    
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

    

# In[7]
traincrnn,predcrnn,inputlength=crnn()
traincrnn.load_weights('weights.300000.h5',by_name=True)
freeze = ['conv1_1',
          'conv2_1',
          'conv3_1', 
          'conv3_2', 
          'conv4_1',
          'conv5_1',
          #'conv6_1',
          #'lstm1',
          #'lstm2'
         ]
for layer in traincrnn.layers:
    layer.trainable=not layer.name in freeze

traincrnn.summary()

# In[8]
inputlength

# In[9]
cropvalpath='D:/engocr/croptestimage/'
croptrainpath='D:/engocr/croptestvalimage/'
cropvaljson='D:/engocr/valcrnn.json'
croptrainjson='D:/engocr/traincrnn.json'
batch=64
traingen=crnnGenerator(imgpath=croptrainpath,labelpath=croptrainjson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
print('trainstart')
traingen.build_data()
print('valstart')
valgen=crnnGenerator(imgpath=cropvalpath,labelpath=cropvaljson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
valgen.build_data()
print('valend')

# In[10]
cropvalpath='D:/engocr/cropvalimage/'
croptrainpath='D:/engocr/croptestimage/'
cropvaljson='D:/engocr/valcrnn.json'
croptrainjson='D:/engocr/valcrnn.json'
batch=64
traingen=crnnGenerator(imgpath=croptrainpath,labelpath=croptrainjson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
print('trainstart')
traingen.build_data()
print('valstart')
valgen=crnnGenerator(imgpath=cropvalpath,labelpath=cropvaljson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
valgen.build_data()
print('valend')

# In[11]
croptrainpath='D:/engocr/cropvalimage/'
cropvalpath='D:/engocr/croptestimage/'
croptrainjson='D:/engocr/valcrnn.json'
cropvaljson='D:/engocr/valcrnn.json'

batch=64
traingen=crnnGenerator(imgpath=croptrainpath,labelpath=croptrainjson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
print('trainstart')
traingen.build_data()
print('valstart')
valgen=crnnGenerator(imgpath=cropvalpath,labelpath=cropvaljson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
valgen.build_data()
print('valend')

# In[12]
croptrainpath='D:/engocr/cropvalimage/'
cropvalpath='D:/engocr/croptestimage/'
croptrainjson='D:/engocr/traincrnn.json'
cropvaljson='D:/engocr/valcrnn.json'

batch=64
traingen=crnnGenerator(imgpath=croptrainpath,labelpath=croptrainjson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
print('trainstart')
traingen.build_data()
print('valstart')
valgen=crnnGenerator(imgpath=cropvalpath,labelpath=cropvaljson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
valgen.build_data()
print('valend')

# In[13]
croptrainpath='D:/engocr/cropvalimage/'
cropvalpath='D:/engocr/croptestimage/'
croptrainjson='D:/engocr/valcrnn.json'
cropvaljson='D:/engocr/traincrnn.json'

batch=64
traingen=crnnGenerator(imgpath=croptrainpath,labelpath=croptrainjson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
print('trainstart')
traingen.build_data()
print('valstart')
valgen=crnnGenerator(imgpath=cropvalpath,labelpath=cropvaljson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
valgen.build_data()
print('valend')

# In[14]
croptrainpath='D:/engocr/cropvalimage/'
cropvalpath='D:/engocr/croptestimage/'
croptrainjson='D:/engocr/valcrnn.json'
cropvaljson='D:/engocr/traincrnn.json'

batch=64
traingen=crnnGenerator(imgpath=croptrainpath,labelpath=croptrainjson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
print('trainstart')
traingen.build_data()
print('valstart')
valgen=crnnGenerator(imgpath=cropvalpath,labelpath=cropvaljson,imgw=256,imgh=32,batch_size=batch,
                      maxlen=62,inputlen=inputlength)
valgen.build_data()
print('valend')

# In[15]
"""
import winsound as sd

def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)
beepsound()
"""
inputlength

# In[16]
traincrnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=0.001,decay=1e-5, clipnorm=1.) ,loss={'ctcloss':lambda y_true,y_pred:y_pred})

# In[17]
from tensorflow.python.keras.callbacks import TensorBoard  ## TensorBoard 를 import합니다.
modelver='crnn_train_v1'
checkdir = './checkpoints/' + datetime.datetime.now().strftime('%Y%m%d%H%M') + '_' + modelver
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())
log_dir = "./logs/fit/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard =TensorBoard(log_dir=log_dir,histogram_freq=1)

# In[18]
q=np.zeros((1,64,128))
len(q[0])

# In[19]
#[image_input,labels,input_length,label_length]
"""
his=traincrnn.fit(x=[valdata,valdata.labels,np.array( inputlengthlist),np.array(vallabellength)],
              y=valdata.labels,
                batch_size=16, 
              epochs=15,
              callbacks=[keras.callbacks.ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True),
                         tensorboard]
             )
"""
his=traincrnn.fit_generator(generator=traingen.next_batch(),
                            steps_per_epoch=int(traingen.imgcount/batch),
                            epochs=30,
                            callbacks=[keras.callbacks.ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, moniter='loss'),
                            tensorboard],
                            validation_data=valgen.next_batch(),
                            validation_steps=int(valgen.imgcount/batch)
                            
                           
                           
                           )

# In[20]
traincrnn,predcrnn,inputlength=crnn()
traincrnn.load_weights('checkpoints/202107191539_crnn_train_v1/weights.005.h5',by_name=True)
freeze = ['conv1_1',
          'conv2_1',
          'conv3_1', 
          'conv3_2', 
          'conv4_1',
          'conv5_1',
          #'conv6_1',
          #'lstm1',
          #'lstm2'
         ]
#for layer in traincrnn.layers:
#    layer.trainable=not layer.name in freeze

traincrnn.summary()

# In[21]
traincrnn,predcrnn,inputlength=crnn()
traincrnn.load_weights('checkpoints/202107191539_crnn_train_v1/weights.005.h5',by_name=True)
freeze = ['conv1_1',
          'conv2_1',
          'conv3_1', 
          'conv3_2', 
          'conv4_1',
          'conv5_1',
          #'conv6_1',
          #'lstm1',
          #'lstm2'
         ]
for layer in traincrnn.layers:
#    layer.trainable=not layer.name in freeze
    layer.trainable=True
traincrnn.summary()

# In[22]
traincrnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=0.001,decay=1e-5, clipnorm=1.) ,loss={'ctcloss':lambda y_true,y_pred:y_pred})

# In[23]
from tensorflow.python.keras.callbacks import TensorBoard  ## TensorBoard 를 import합니다.
modelver='crnn_train_v1'
checkdir = './checkpoints/' + datetime.datetime.now().strftime('%Y%m%d%H%M') + '_' + modelver
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())
log_dir = "./logs/fit/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard =TensorBoard(log_dir=log_dir,histogram_freq=1)

