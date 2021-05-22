import tensorflow as tf

import tensorflow.keras.layers as nn
import numpy as np
import cv2
print(tf.executing_eagerly())

from netVLADlayer import netVLADlayer
from tensorflow.keras.applications import MobileNetV2
from detector import Detector
from descriptor import Descriptor
from Loss import *
#import netvlad and superpoint

class HFnet(tf.keras.Model) :
    def __init__(self, in_shape=(640,480,3),alpha=0.75) :
        super(HFnet,self).__init__()

        #self.input_shape=input_shape
        self.alpha=alpha
        self.block_size=8
        self.backbone=MobileNetV2(input_shape=in_shape,alpha=self.alpha, include_top=False,weights='imagenet')
        # self.backbone.summary()

        # self.hidden=self.backbone.get_layer('block_6_project').output#pass layer name
        self.detector_head=Detector() #detector head
        self.descriptor_head=Descriptor() #descriptor head
        self.netvladlayer=netVLADlayer(dim = 1280) #netvladlayer

        # self.loss_multipliers=tf.Variable(shape=(3,1),
        #                 initial_value=tf.ones(shape=(3,1)))#Loss multipliers
        # self.compiled_loss = Loss()
        # self.optimizer = tf.keras.optimizers.RMSprop()


    def call(self,x,training=None) :
        x=self.backbone(x)
        # print(x.shape)
 
        global_descriptor=self.netvladlayer(x)
        
        local_descriptor=self.descriptor_head(self.hidden)
        key_points=self.detector_head(self.hidden)

        #if training :
        return [global_descriptor,local_descriptor,key_points]#,self.loss_multipliers

        #throw away the dustbin and return probabilistic scores
        # key_points_map=tf.nn.depth_to_space(key_points[:,:,:,:-1],self.block_size)
        # return [global_descriptor,local_descriptor,key_points_map, self.loss_multipliers]

    '''
    def train_step(self, data):
        x, y = data
        # print('**************')
        # print(len(y))
        with tf.GradientTape() as tape:
            y_pred = self(x, training = True)
            # print('#########**************')
            # print(len(y_pred))
            train_loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(train_loss, self.trainable_variables)
    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if self.step%100==0:
            y_val_pred = self(x_val, training = False)
            val_loss =self.compiled_loss(y_val, y_val_pred)
        print(val_loss)

        if self.step == 60000:
            optimizer.learning_rate.assign(0.0001)
        if self.step==80000:
            optimizer.learning_rate.assign(0.00001)
        self.step = self.step + 1
      
'''

model = HFnet(in_shape = (45, 45, 3))

x = tf.random.normal((2, 45,45,3))
x1=tf.random.normal(shape=(2,4096))
x2=tf.random.normal(shape=(2,3,3,256))
x3=tf.random.normal(shape=(2,3, 3, 65))
# x4 = tf.random.normal(shape = (2, 3, 1))
y = [x1, x2, x3]

# y_pred = model(x)
# print(len(y_pred))
# print(y_pred[1].shape)
# print(y_pred[2].shape)

# print(len(y))
# print(loss(y, y_pred))
# print(tf.__version__)
model.compile(optimizer = "RMSprop", 
	loss = [Loss_desc('gdesc'),Loss_desc('ldesc'),Loss_ldet()])

# model.train_step((x, y))
model.fit(x, y, epochs = 1, batch_size = 2)

