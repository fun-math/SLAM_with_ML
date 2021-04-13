import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

from ConvBlock import *

class VGG(tf.keras.Model) :
	def __init__(self,in_shape=(224,224,3)) :
		super(VGG,self).__init__()
		self.in_shape=in_shape
		self.conv_build=nn.Conv2D(64,3,1,padding='same',activation='relu',input_shape=(224,224,3))
		self.block1=ConvBlock(1,64,in_shape=self.in_shape)
		self.block2=ConvBlock(2,128)
		self.block3=ConvBlock(3,256)
		self.block4=ConvBlock(3,512)
		self.block5=ConvBlock(3,512,logits=True)
		self.norm=nn.Lambda(lambda x : tf.math.l2_normalize(x,axis=-1))

	def call(self,x) :
		x=self.conv_build(x)
		x=self.block1(x)
		x=self.block2(x)
		x=self.block3(x)
		x=self.block4(x)
		x=self.block5(x)
		x=self.norm(x)

		return x

if __name__=='__main__':
	vgg=VGG()
	vgg.build((None,224,224,3))
	vgg.summary()
