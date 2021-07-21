#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


import cv2
import copy
import string


# In[3]:


def imagecreate(imgpath,gray=False):
    imgw=320
    imgh=320
    if gray==False:
        img=cv2.imread(imgpath)
    else:
        img=cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    img=img.astype(np.float32)
    img=cv2.resize(img,(imgw,imgh))

    cv2.imwrite('abc.jpg',img)
    img=img/255.0
    return img
def yoloimagecreate(imgpath):

    img=imagecreate(imgpath)
    img=np.expand_dims(img,0)
    return img
#yolo랑 crnn이랑 묶어서만들어도 될거같긴한데 귀찮아..


# In[4]:


def yolopostprocess(features,threshold=0.0):
    def sigmoid(x,threshold=0.0):
        return (1./(1.+np.exp(-x)))+threshold
    #교체할일있을때 앵커랑 앵커마스크,클래스수 하드코딩말고 밖으로 빼는거도 생각
    num_classes=1
    sigmoid = np.vectorize(sigmoid)
    proto_box=[]
    proto_scores=[]
    for idx, val in enumerate(features):
        #교체할일있을때 앵커랑 앵커마스크,클래스수 하드코딩말고 밖으로 빼는거도 생각
        anchors = np.array([np.array([10,13]), np.array([16,30]), np.array([33,23]), np.array([30,61]), np.array([62,45]), np.array([59,119]), np.array([116,90]), np.array([156,198]), np.array([373,326])])
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        input_shape = np.asarray(np.shape(features[0])[1 : 3]) * 32
        first = anchors[anchor_mask[idx]]
        image_size = (320, 320)
        num_anchors = len(first)
        anchors_tensor = np.reshape(first, [1, 1, 1, num_anchors, 2])
        grid_shape = np.shape(val)[1 : 3]
        b = np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1])
        grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
        grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                        [grid_shape[0], 1, 1, 1])
        grid = np.concatenate([grid_x, grid_y], axis=3)
        feats = np.reshape(
                val, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        box_xy = (sigmoid(feats[..., :2],threshold) + grid) / grid_shape[::-1]
        pre_box_wh = feats[..., 2:4] * anchors_tensor / input_shape[::-1]
        box_wh = np.exp(feats[..., 2:4]) * anchors_tensor / input_shape[::-1]
        box_confidence = sigmoid(feats[..., 4:5],threshold)
        box_class_probs = sigmoid(feats[..., 5:],threshold)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        image_shape = np.array([320, 320])
        new_shape = np.round((image_shape * np.min(input_shape/image_shape)))
        offset = (input_shape-new_shape)/2./input_shape
        scale = input_shape/new_shape
        box_yx = (box_yx - offset) #* scale
        #box_hw *= scale
        #box_hw*=1.0

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        boxes = np.concatenate([
                box_mins[..., 0:1],  # y_min
                box_mins[..., 1:2],  # x_min
                box_maxes[..., 0:1],  # y_max
                box_maxes[..., 1:2]  # x_max
            ], axis=4)

        # Scale boxes back to original image shape.
        scaler = np.concatenate([image_shape, image_shape])
        boxes *= scaler
        #here at original implementation is loosing of data, because batch size is ignored
        boxes = np.reshape(boxes, [boxes.shape[0], -1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = np.reshape(box_scores, [box_scores.shape[0], -1, num_classes])
        proto_box.append(boxes)
        proto_scores.append(box_scores)
    proto_box = np.concatenate(proto_box, axis=1)
    proto_scores = np.concatenate(proto_scores, axis=1)
    mask = proto_scores >= 0.6
    _boxes = []
    #yolo모델후처리(함수로빼자)    
    for idx, batch in enumerate(proto_scores):
        final_classes = []
        final_boxes = []
        final_scores = []
        for c in range(num_classes):
            class_boxes = proto_box[idx, mask[idx, :, c]]
            class_box_scores = proto_scores[idx, :, c][mask[idx, :, c]]
            classes = np.ones_like(class_box_scores, dtype="int32") * c
            final_boxes.append(class_boxes)
            final_scores.append(class_box_scores)
        final_boxes = np.concatenate(final_boxes, axis=0)
        final_scores = np.concatenate(final_scores, axis=0)
        _boxes.append(final_boxes)
    #yolo모델후처리(함수로빼자 위랑묶어서)
    return final_boxes


# In[5]:


def crnnpredict(model,imgpath,boxes):
    img=imagecreate(imgpath,gray=True)
    outputdata=[]
    for cropbox in boxes:
        cropped_img = copy.deepcopy(img[int(cropbox[0]):  int(cropbox[2]),int(cropbox[1]): int(cropbox[3])])
        cropped_img=cv2.resize(cropped_img,(256,32))
        cropped_img=cropped_img.astype(np.float32)        
        cropped_img=cropped_img.T
        cropped_img=np.expand_dims(cropped_img,axis=-1)
        cropped_img=np.expand_dims(cropped_img,axis=0)
        outputdata.append (model.predict(cropped_img))
    return outputdata


# In[6]:


def labels_to_text(labels):     
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    alphabet87 = string.ascii_lowercase + string.ascii_uppercase + string.digits + ' +-*.,:!?%&$~/()[]<>"\'@#_'
    return ''.join(list(map(lambda x: alphabet87[int(x)], labels)))


# In[7]:


def postcrnn(outputdata):
    rtrlabel=[]
    for i,out in enumerate( outputdata):
        res=[]
        for data in out[0]:
            listdata=list(data)
            res.append( listdata.index(max(listdata)))

        resu=[]
        for i,abc in enumerate(res) :
            if abc!=86:
                resu.append(abc)
        rtrlabel.append(labels_to_text(resu))
    return rtrlabel


# In[ ]:




