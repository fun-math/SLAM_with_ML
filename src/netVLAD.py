import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

from VGG import *
from netVLADlayer import *

class netVLAD(tf.keras.Model) :
	def __init__(self,):
		super(netVLAD,self).__init__()
		self.vgg=VGG()
		self.netVLADlayer=netVLADlayer()

	def call(self,x):
		x=self.vgg(x)
		x=self.netVLADlayer(x)

		return x

if __name__=='__main__':
	m=netVLAD()
	m.build((1,224,224,3))
	m.summary()
	# m.summary()