#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
from yolo3.model_Mobilenet import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


from keras.layers import Dense,Reshape,Activation,BatchNormalization,Lambda,Input,Conv2D,Bidirectional,LSTM, Concatenate, MaxPooling2D,GlobalAveragePooling2D


# In[2]:


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l],         num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    print(model_body.output)
    #model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
    #    arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
    #    [*model_body.output, *y_true])
    model = Model([model_body.input], outputs=[model_body.get_layer('conv2d_6').output,model_body.get_layer('conv2d_14').output,model_body.get_layer('conv2d_22').output,] )

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l],         num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


# In[3]:


def createyolo(weights_path='D:/engocr/outputmodel/yolomodel/yoloweight.h5'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_path = 'train.txt'
    val_path = 'val.txt'
    log_dir = 'logs/carMobilenet/001_Mobilenet_finetune/'
    classes_path = 'model_data/car_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (320,320) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2)
    else:
        model = create_model(input_shape, anchors, num_classes,load_pretrained=False,
                             weights_path=weights_path,
            freeze_body=2) # make sure you know what you freeze
        print('a')

    logging = TensorBoard(log_dir=log_dir)
    # checkpoint = ModelCheckpoint(log_dir + 'car_mobilenet_yolov3.ckpt',
    #    monitor='val_loss', save_weights_only=False, period=1)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    return model


# In[8]:


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
    x=Reshape((s[1],s[3]),name='reshape_1')(x)#32 2048
    x=Bidirectional(LSTM(256, return_sequences=True,name='bidirectional_1'))(x)
    x=Bidirectional(LSTM(256, return_sequences=True,name='bidirectional_1'))(x)#32 512
    
    num_classes=87
    x=Dense(num_classes, name='dense1')(x)#알파벳+숫자#1953 36
    
    x=y_pred= Activation('softmax', name='softmax')(x)#1953 36
    
   # model_pred=keras.models.Model(inputs=[image_input,minx,miny,maxx,maxy],outputs=x)
    model_pred=Model(inputs=image_input,outputs=x)
    
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
    model_train=Model(inputs=[image_input,labels,input_length,label_length],outputs=ctcloss)
    
    return model_pred


# In[6]:





# In[ ]:




