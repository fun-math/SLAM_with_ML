import tensorflow as tf

import tensorflow.keras.layers as nn
import numpy as np
import cv2

from netVLADlayer import netVLADlayer
from detector import Detector
from descriptor import Descriptor
from Loss import *
from MobileNetv2 import *

class HFnet(tf.keras.Model) :
    def __init__(self, in_shape=(640,480,3),alpha=0.75,mid=7) :
        super(HFnet,self).__init__()

        self.in_shape=in_shape
        self.alpha=alpha
        self.block_size=8
        self.base=MobileNetV2(input_shape=in_shape,include_top=False,alpha=0.75,
                weights='../weights/custom_mobilenet_v2_0.75_224_no_top.h5',
                begin=True,start_block=0,finish_block=mid)
        self.netvlad_encoder=MobileNetV2(input_shape=in_shape,include_top=False,alpha=0.75,
                weights='../weights/custom_mobilenet_v2_0.75_224_no_top.h5',
                end=True,start_block=mid+1,finish_block=16)

        self.detector_head=Detector() #detector head
        self.descriptor_head=Descriptor() #descriptor head
        self.netvladlayer=netVLADlayer(dim = 1280) #netvladlayer
        self.step=0


    def call(self,x,training=None) :
        x=self.base(x)
 
        global_descriptor=self.netvladlayer(self.netvlad_encoder(x))
        
        local_descriptor=self.descriptor_head(x)
        key_points=self.detector_head(x)

        if training :
            return [global_descriptor,local_descriptor,key_points]

        #throw away the dustbin and return probabilistic scores
        key_points_map=tf.nn.depth_to_space(key_points[:,:,:,:-1],self.block_size)
        return [global_descriptor,local_descriptor,key_points_map]

    
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training = True)
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

    def build_graph(self) :
        x=tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x],outputs=self.call(x))
      

if __name__=='__main__' :

    model = HFnet(in_shape = (640, 480, 3),mid=7)
    model.build((None,640,480,3))
    # tf.keras.utils.plot_model(model.build_graph(),to_file='HFnet.png',expand_nested=True)

    x = tf.random.normal((2, 640,480,3))
    x1=tf.random.normal(shape=(2,4096))
    x2=tf.random.normal(shape=(2,40,30,256))
    x3=tf.random.normal(shape=(2,40, 30, 65))
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

    # # model.train_step((x, y))
    model.fit(x, y, epochs = 1, batch_size = 2)

