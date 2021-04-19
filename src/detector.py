import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

class Detector(tf.keras.layers.Layer) :
	def __init__(self) :
		super(Detector,self).__init__()
		self.conv1=nn.Conv2D(256,3,1,padding='same',activation='relu')
		self.conv2=nn.Conv2D(65,1,1,padding='same')
		self.softmax=nn.Softmax()

	def call(self,x) :
		x=self.conv1(x)
		x=self.conv2(x)
		x=self.softmax(x)

		return x

if __name__=='__main__' :
	desc=Detector()
	x=tf.random.normal(shape=(1,120,120,96))
	y=desc(x)
	print(y.shape)