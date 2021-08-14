import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

from ConvBlock import *

class VGG(tf.keras.Model) :
	def __init__(self) :
		super(VGG,self).__init__()
		self.block1=ConvBlock(2,64)
		self.block2=ConvBlock(2,128)
		self.block3=ConvBlock(3,256)
		self.block4=ConvBlock(3,512)
		self.block5=ConvBlock(3,512,logits=True)
		self.norm=nn.Lambda(lambda x : tf.math.l2_normalize(x,axis=-1))

	def call(self,x) :
		x=self.block1(x)
		x=self.block2(x)
		x=self.block3(x)
		x=self.block4(x)
		x=self.block5(x)
		x=self.norm(x)

		return x

if __name__=='__main__':
	vgg=VGG()
	# vgg.build((None,224,224,3))
	x=tf.random.normal(shape=(1,224,224,3))
	y=vgg(x)
	print(y.shape)
