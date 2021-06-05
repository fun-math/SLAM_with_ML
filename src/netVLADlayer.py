import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

class netVLADlayer(tf.keras.layers.Layer) :
	def __init__(self,num_clusters=64,dim=512,weight_init='glorot_uniform',cluster_initializer=None,postnorm=True) :
		super(netVLADlayer,self).__init__()
		self.postnorm=postnorm

		self.conv1=nn.Conv2D(num_clusters,1,1,kernel_initializer=weight_init)
		self.softmax=nn.Softmax()
		self.C=tf.Variable(initial_value=tf.random.uniform([1,1,1,dim,num_clusters]),shape=[1,1,1,dim,num_clusters])
		self.vec1=nn.Flatten()
		self.conv2=nn.Conv2D(4096,1,1)
		self.vec2=nn.Flatten()

	def call(self,x) :
		s=self.conv1(x)
		s=self.softmax(s)
		a=tf.expand_dims(s,-2)

		v=tf.expand_dims(x,-1)+self.C
		v=a*v
		v=tf.math.reduce_sum(v,axis=[1,2])
		v=tf.transpose(v,perm=[0,2,1])

		if self.postnorm :
			v=tf.math.l2_normalize(v,axis=-1)
			v=tf.math.l2_normalize(self.vec1(v),axis=-1)

		#PCA (rather PCA a simple 1 by 1 convolution for dimensionality reduction)
		v=tf.expand_dims(v,1)
		v=tf.expand_dims(v,1)
		v=self.conv2(v)
		v=tf.math.l2_normalize(self.vec2(v))
        
		return v

if __name__=='__main__':
	vlad=netVLADlayer(dim=512)
	x=tf.random.normal(shape=(1,7,7,512))
	y=vlad(x)
	print(y.shape)