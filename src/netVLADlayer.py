import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

class netVLADlayer(tf.keras.layers.Layer) :
	def __init__(self,num_clusters=64,dim=128,weight_init='glorot_uniform',cluster_initializer=None,postnorm=True) :
		super(netVLADlayer,self).__init__()
		# self.num_clusters=num_clusters
		# self.cluster_initializer=cl
		self.postnorm=postnorm

		self.conv1=nn.Conv2D(num_clusters,1,1,kernel_initializer=weight_init)
		self.softmax=nn.Softmax()
		self.C=tf.Variable(initial_value=tf.random.uniform([1,1,1,dim,num_clusters]),shape=[1,1,1,dim,num_clusters])
		self.vec=nn.Flatten()

	def call(self,x) :
		x=self.conv1(x)

		s=self.softmax(x)
		a=tf.expand_dims(s,-2)

		v=tf.expand_dims(x,-1)+self.C
		v=a*v
		v=tf.math.reduce_sum(v,axis=[1,2])
		v=tf.transpose(v,perm=[0,2,1])

		if self.postnorm :
			v=tf.math.l2_normalize(v,axis=-1)
			v=tf.math.l2_normalize(self.vec(v),axis=-1)

		return v
