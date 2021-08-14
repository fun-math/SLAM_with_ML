import tensorflow as tf 
import numpy as np 

class Loss_desc(tf.keras.losses.Loss) :
	def __init__(self,name=None) :
		super(Loss_desc,self).__init__(name=name)
		self.loss_multiplier=tf.Variable(shape=(),initial_value=1.0)

	def call(self,y_true,y_pred) :
		loss_desc=tf.math.reduce_mean(
					tf.math.reduce_sum(
						tf.math.square(y_true-y_pred),axis=-1))

		return (loss_desc+1)*self.loss_multiplier


class Loss_ldet(tf.keras.losses.Loss) :
	def __init__(self,name='ldet') :
		super(Loss_ldet,self).__init__(name=name)
		self.loss_multiplier=tf.Variable(shape=(),initial_value=1.0)

	def call(self,y_true,y_pred) :
		loss_ldet=tf.math.reduce_mean(
					tf.keras.losses.categorical_crossentropy(y_true,y_pred))

		return (loss_ldet+1)*self.loss_multiplier


class Loss(tf.keras.losses.Loss) :
	def __init__(self) :
		super(Loss,self).__init__(name='loss')
		self.loss_multipliers=tf.Variable(shape=(),initial_value=1)

	def call(self,y_true,y_pred) :
		loss_multipliers=y_pred[-1]
		coeffs=tf.math.exp(loss_multipliers)
		print(y_true.shape,y_pred.shape)
		print('------------')
		'''
		for i in range(4) :
			print(i,y_true[i].shape,y_pred[i].shape)
		loss_gdesc=tf.math.reduce_mean(
					tf.math.reduce_sum(
						tf.math.square(y_true[0]-y_pred[0]),axis=-1))
		loss_ldesc=tf.math.reduce_mean(
					tf.math.reduce_sum(
						tf.math.square(y_true[1]-y_pred[1]),axis=-1))
		loss_ldet=tf.math.reduce_mean(
					tf.keras.losses.categorical_crossentropy(y_true[2],y_pred[2]))

		loss=coeffs[0]*loss_gdesc+coeffs[1]*loss_ldesc+2*coeffs[2]*loss_ldet+tf.math.reduce_sum(loss_multipliers)
		'''
		return 0#loss


if __name__=='__main__' :
	
	loss_gdesc=Loss_desc('gdesc')
	loss_ldesc=Loss_desc('ldesc')
	loss_ldet=Loss_ldet()
	x1=tf.random.normal(shape=(3,4096))
	y1=tf.random.normal(shape=(3,4096))
	x2=tf.random.normal(shape=(3,80,80,64))
	y2=tf.random.normal(shape=(3,80,80,64))
	x3=tf.random.normal(shape=(3,80,80,65))
	y3=tf.random.normal(shape=(3,80,80,65))
	
	print(loss_gdesc(x1,y1),loss_ldesc(x2,y2),loss_ldesc(x3,y3))