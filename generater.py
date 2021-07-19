#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random
import numpy as np
import cv2
import json 
import pandas as pd


# In[2]:


def convert_to_lable(data):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return list(map(lambda x: alphabet.index(x), data))
convert_to_lable('afsfs')


# In[7]:


class crnnGenerator:
    def __init__(self,imgpath,labelpath,imgw,imgh,batch_size,maxlen,inputlen):
        self.imgw=imgw
        self.imgh=imgh
        self.batch_size=batch_size
        self.imgpath=imgpath
        self.imgdir=os.listdir(self.imgpath)
        self.imgcount=len(self.imgdir)
        self.index=list(range(self.imgcount))
        self.curindex=0
        self.maxlen=maxlen
        self.imgs=np.zeros((self.imgcount,self.imgh,self.imgw))
        self.texts=[]
        self.labelpath=labelpath
        self.labelpd=pd.read_json(self.labelpath,orient='index')
        self.inputlen=inputlen
    
    def build_data(self):
        for i,file in enumerate(self.imgdir):
            a=self.labelpd.loc[self.labelpd['cropimgid']==file[0:-4] ]['label']
            if (len(a.values[0])) <self.maxlen:
                img=cv2.imread(self.imgpath+file,cv2.IMREAD_GRAYSCALE)
                img=cv2.resize(img,(self.imgw,self.imgh))
                img=img.astype(np.float32)
                img=(img / 255.0) * 2.0 - 1.0

                self.imgs[i,:,:]=img

                


                self.texts.append(convert_to_lable(a.values[0]))
            else:
                self.imgcount=self.imgcount-1
        self.index=list(range(self.imgcount)) 
    def next_sample(self):
        self.curindex+=1
        if self.curindex>=self.imgcount:
            self.curindex=0
            random.shuffle(self.index)
        
        a=self.imgs[self.index[self.curindex]]
        b=self.texts[self.index[self.curindex]]
        return a,b
    
    def test(self,i=1,j=2):
        a=self.imgs[self.index[self.curindex]]
        b=self.texts[self.index[self.curindex]]
        return a,b
    
    def texttest(self,file='000adfe5b817011c_1'):
        img=cv2.imread(self.imgpath+file+'.jpg',cv2.IMREAD_GRAYSCALE)

        img=cv2.resize(img,(self.imgw,self.imgh))
        img=img.astype(np.float32)
        img=img/255.0
        cv2.imshow('ada',img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def next_batch(self):
        while True:
            xdata=np.ones([self.batch_size,self.imgw,self.imgh,1])
            ydata=np.ones([self.batch_size,self.maxlen])
            inputlength=np.ones((self.batch_size,1))*self.inputlen   #(self.imgw//2) 최대길이
            labellength=np.zeros((self.batch_size,1))
            
            for i in range(self.batch_size):
                img,text=self.next_sample()
                
                img=img.T
                img=np.expand_dims(img,-1)
                xdata[i]=img
                for j,k in enumerate(text):
                    
                    
                    ydata[i][j]=k
                labellength[i]=len(text)
                
            inputs={
                'image_input':xdata,
                'labels':ydata,
                'input_length':inputlength,
                'label_length':labellength
            }
            outputs={'ctcloss':np.zeros([self.batch_size])}
            yield(inputs,outputs)
            
            
        
        


# In[5]:


"""
cropvalpath='D:/engocr/abcd/'
croptrainpath='D:/engocr/abcd/'
cropvaljson='D:/engocr/valcrnn.json'
batch=32
traingen=crnnGenerator(imgpath=croptrainpath,labelpath=cropvaljson,imgw=128,imgh=64,batch_size=batch,
                      maxlen=100,inputlen=100)
print('trainstart')
traingen.build_data()
print('valstart')
valgen=crnnGenerator(imgpath=cropvalpath,labelpath=cropvaljson,imgw=128,imgh=64,batch_size=batch,
                      maxlen=100,inputlen=100)
valgen.build_data()
print('valend')

"""


# In[6]:


#valgen.test()
#valgen.texttest()


# In[7]:


#ada=cv2.resize(img,(self.imgw,self.imgh))


# In[36]:


"""
y = np.ones([16, 100])
b=convert_to_lable('afsfs')
for i,j in enumerate(b):
    print(i,j)
    y[1][i]=j
y
"""


# In[38]:





# In[ ]:




