import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
import cv2

from tf.keras.applications import MobileNetV2
from detector import Detector
from descriptor import Descriptor
#import netvlad and superpoint

class HFnet(tf.keras.Model) :
	def __init__(input_shape=(640,480,3),alpha=0.75) :
		super(HFnet,self).__init__()

		self.input_shape=input_shape
		self.alpha=alpha
		self.block_size=8
		self.backbone=MobileNetV2(input_shape=self.input_shape,alpha=self.alpha,
						include_top=False,weights='imagenet')
		self.hidden=backbone.get_layer()#pass layer name
		self.detector_head=Detector() #detector head
		self.descriptor_head=Descriptor() #descriptor head
		self.netvladlayer=None #netvladlayer

		self.loss_multipliers=tf.Variable(shape=(3,1),
						initial_value=tf.ones(shape=(3,1)))#Loss multipliers


	def call(self,x,training=None) :
		x=self.backbone(x)

		global_descriptor=self.netvladlayer(x)
		local_descriptor=self.descriptor_head(self.hidden)
		key_points=self.detector_head(self.hidden)

		if training :
			return [global_descriptor,local_descriptor,key_points,self.loss_multipliers]

		#throw away the dustbin and return probabilistic scores
		key_points_map=tf.nn.depth_to_space(np.array(key_points)[:,:,:,:-1],self.block_size)
		return [global_descriptor,local_descriptor,key_points_map]
