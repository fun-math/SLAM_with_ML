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
    def __init__(self, in_shape=(480,640,3),alpha=0.75,mid=5,weights_dir=None) :
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
        

        self.weights_dir=weights_dir
        self.step=0#None
        self.valid_freq=None
        self.train_ds=None
        self.valid_ds=None
        self._losses=None
        self.optimizer=tf.keras.optimizers.RMSprop()




    def call(self,x,training=None) :
        x=self.base(x)
 
        global_descriptor=self.netvladlayer(self.netvlad_encoder(x))
        
        local_descriptor=self.descriptor_head(x)
        key_points=self.detector_head(x)

        # if training :
        return [global_descriptor,local_descriptor,key_points]

        #throw away the dustbin and return probabilistic scores
        # key_points_map=tf.nn.depth_to_space(key_points[:,:,:,:-1],self.block_size)
        # return [global_descriptor,local_descriptor,key_points_map]

    
    def assign_data(self,train_ds=None,valid_ds=None) :
        self.train_ds=train_ds
        self.valid_ds=valid_ds

    def assign_loss(self,_losses) :
        ''' _losses=[loss1,loss2,loss3]'''
        self._losses=_losses

    def calculate_loss(self,y,y_pred) :
        return sum([self._losses[i](y[i],y_pred[i]) for i in range(3)])

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training = True)
            train_loss = self.calculate_loss(y, y_pred)

        gradients = tape.gradient(train_loss, self.trainable_variables)
    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # self.compiled_metrics.update_state(y, y_pred)
        # if self.step%100==0:
        #     for x_batch_val, y_batch_val in val_dataset:
        #         y_batch_val_pred = self(x_batch_val, training = False)
        #         val_loss =self.compiled_loss(y_val, y_val_pred)
        # print(val_loss)

        if self.step == 60000:
            optimizer.learning_rate.assign(0.0001)
        if self.step==80000:
            optimizer.learning_rate.assign(0.00001)
        self.step = self.step + 1
        # print(train_loss.numpy())
        return train_loss
        # tf.print(train_loss)
        # print(self.metrics[0].result())
        # for m in self.metrics :
        #     tf.print(m.result()) 
        # return {m.name: m.result() for m in self.metrics}
        
    def test_step(self,data) : 
        x, y = data
        y_pred=self(x)
        #self.compiled_metrics.update_state(y,y_pred)
        loss = self.calculate_loss(y, y_pred)
        return loss

    def custom_fit(self,steps = 85000,valid_freq=100) :
        epochs=np.ceil(steps/self.train_ds.__len__())
        self.valid_freq=valid_freq
        self.step=0

        for epoch in range(int(epochs)) :
            mean_loss = 0
            for x_batch_train, y_batch_train in self.train_ds :
                loss=self.train_step((x_batch_train, y_batch_train))
                loss_numpy = loss.data.cpu().numpy()
                print('step = ',self.step,', loss = ',loss_numpy)

                if self.step % self.valid_freq == 0 :
                    for x_batch_val, y_batch_val in self.valid_ds :
                        val_loss  = self.test_step((x_batch_val, y_batch_val))
                        val_loss_numpy = val_loss.data.cpu().numpy()
                    print('**********Validation*********\nloss = ',val_loss_numpy,
                        '\n*****************************')
                    #for m in self.metrics :
                        #m.reset_states()

                    # self.save_weights(self.weights_dir+'hfnet_last.h5')


    def build_graph(self) :
        x=tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x],outputs=self.call(x))
      

if __name__=='__main__' :

    model = HFnet(in_shape = (160, 160, 3),mid=7)
    model.build((None,160,160,3))
    # tf.keras.utils.plot_model(model.build_graph(),to_file='HFnet.png',expand_nested=True)

    x = tf.random.normal((2, 160,160,3))
    x1=tf.random.normal(shape=(2,4096))
    x2=tf.random.normal(shape=(2,10,10,256))
    x3=tf.random.normal(shape=(2,10, 10, 65))
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

