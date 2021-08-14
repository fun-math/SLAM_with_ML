import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

class Descriptor(tf.keras.layers.Layer) :
	def __init__(self,dim=256) :
		super(Descriptor,self).__init__()
		self.dim=dim
		self.conv1=nn.Conv2D(256,3,1,padding='same',activation='relu')
		self.conv2=nn.Conv2D(self.dim,1,1,padding='same')

	def call(self,x) :
		x=self.conv1(x)
		x=self.conv2(x)
		x=tf.math.l2_normalize(x,axis=-1)

		return x

if __name__=='__main__' :
	desc=Descriptor()
	x=tf.random.normal(shape=(1,120,120,96))
	y=desc(x)
	print(y.shape)