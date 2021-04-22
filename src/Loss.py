import tensorflow as tf 
import numpy as np 
import tf.random.normal as randnorm

class Loss(tf.keras.losses.Loss) :
	def __init__(self) :
		pass

	def call(y_true,y_pred) :
		loss_multipliers=y_pred[-1]
		coeffs=tf.math.exp(loss_multipliers)

		loss_gdesc=tf.math.reduce_sum(tf.math.square(y_true[0]-y_pred[0]),axis=-1)
		loss_ldesc=tf.math.reduce_sum(tf.math.square(y_true[1]-y_pred[1]),axis=-1)
		loss_ldet=tf.keras.losses.categorical_crossentropy(y_true[2],y_pred[2])

		loss=coeffs*np.array([loss_gdesc,loss_ldesc,loss_ldet])+tf.math.reduce_sum(loss_multipliers)

		return loss

if __name__=='__main__' :
	loss=Loss()
	x1=randnorm(shape=(3,4096))
	y1=randnorm(shape=(3,4096))
	x2=randnorm(shape=(3,80,80,64))
	y2=randnorm(shape=(3,80,80,64))
	x3=randnorm(shape=(3,80,80,65))
	y3=randnorm(shape=(3,80,80,65))
	mult=randnorm(shape=(3,1))
	l=loss([x1,x2,x3],[y1,y2,y3,mult])
	print(l)