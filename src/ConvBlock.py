import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

class ConvBlock(tf.keras.layers.Layer) :
	def __init__(self, layers, channels,logits=False,in_shape=None) :
		super(ConvBlock, self).__init__()
		self.layers=layers
		self.channels=channels
		self.logits=logits
		self.convs=[]
		for i in range(self.layers-1) :
			self.convs+=[nn.Conv2D(self.channels,3,1,padding='same',activation='relu')]
		self.convs+=[nn.Conv2D(self.channels,3,1,padding='same')]
		# if in_shape!=None :
		# 	self.convs[0]=nn.Conv2D(self.channels,3,1,padding='same',activation='relu',input_shape=in_shape)
		self.pool=nn.MaxPool2D(2,2)
		if not self.logits :
			self.relu=nn.ReLU()

	def call(self,x) :
		for i in range(self.layers) :
			x=self.convs[i](x)
		x=self.pool(x)
		if not self.logits :
			x=self.relu(x)
		return x

if __name__=='__main__' :
	c=ConvBlock(3,64)
	x=tf.random.normal(shape=(1,224,224,3))
	y=c(x)
	print(y)