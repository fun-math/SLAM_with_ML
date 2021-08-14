import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

class Detector(tf.keras.layers.Layer) :
	def __init__(self,axis=-1) :
		super(Detector,self).__init__()
		self.conv1=nn.Conv2D(256,3,1,padding='same',activation='relu')
		self.conv2=nn.Conv2D(65,1,1,padding='same')
		self.axis = axis

	def call(self,x) :
		x=self.conv1(x)
		x=self.conv2(x)
		x=self.Softmax(x)

		return x


	def Softmax(self,x) :
		m = tf.math.reduce_max(x, axis = self.axis, keepdims = True)
		lse = m + tf.math.log(tf.math.reduce_sum(tf.math.exp(x - m), axis = self.axis, keepdims = True))
		soft = tf.math.exp(x - lse)

		return soft
    


if __name__=='__main__' :
	desc=Detector()
	x=10*tf.random.normal(shape=(1,120,120,96))
	y=desc(x)
	print(y.shape)
	print(tf.norm(np.sum(y,axis=-1)-np.ones(y.shape[:3]))/tf.norm(np.ones(y.shape[:3])))
