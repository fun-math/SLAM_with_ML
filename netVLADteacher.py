import cv2
import numpy as np
import tensorflow as tf
import glob
from time import time

import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets

tf.reset_default_graph()

image_batch = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3])

net_out = nets.vgg16NetvladPca(image_batch)
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, nets.defaultCheckpoint())

inim = cv2.imread(nfm.exampleImgPath())
inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)

batch = np.expand_dims(inim, axis=0)
result = sess.run(net_out, feed_dict={image_batch: batch})

images_path='/media/ironwolf/students/amit/datasets/bdd100k/images/100k/'
labels_path='/media/ironwolf/students/amit/datasets/bdd100k/labels/100k/'

for dataset in ['train/','val/','test/'] :
	names=glob.glob(images_path+dataset+'*')
	for name in names :
		img=cv2.imread(name)
		img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		batch=np.expand_dims(img,axis=0)
		tic=time()
		result = sess.run(net_out, feed_dict={image_batch: batch})
		toc=time()
		print(toc-tic)
		# np.save(labels_path+dataset+name.split('/')[-1][:-3]+'npy',result)
		print(labels_path+dataset+name.split('/')[-1][:-3]+'npy')