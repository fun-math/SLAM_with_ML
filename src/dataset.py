import tensorflow as tf
import numpy as np
import cv2
import glob

#/media/ironwolf/students/amit/datasets/bdd100k/images/100k/train/
#/media/ironwolf/students/amit/datasets/bdd100k/labels/100k/<label_type>/train/
# <label_type> is one of ['gdesc/','ldesc/','ldet/']

class Dataset(tf.keras.utils.Sequence) :
	def __init__(self,pre_path="/media/ironwolf/students/amit/datasets/bdd100k/",
		post_path='100k/',split='train/',batch_size=64) :
		
		self.pre_path=pre_path
		self.post_path=post_path
		self.split=split
		self.batch_size=batch_size
		self.label_types=[ 'ldesc/','ldet/']
		self.names=[full_name.split('/')[-1] 
			for full_name in glob.glob(pre_path+'images/'+post_path+split+'*')]

	def __len__(self) :
		#random value
		# return 4

		#correct output
		return np.ceil(len(self.names)/self.batch_size)

	def _imread(self,path) :
		
		#print(path)
		img=cv2.imread(path, 0)
		img = cv2.resize(img, (640, 480))
		img = np.expand_dims(img, axis = -1) 
		img = np.repeat(img, 3, axis = -1).astype(np.float32)
		#print(img.shape)
		#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		return img

	def __getitem__(self,idx) :
		batch_names=self.names[idx*self.batch_size : (idx+1)*self.batch_size]

		batch_x=np.array([self._imread('{}images/{}{}{}'.format(self.pre_path,self.post_path,self.split,name))
				for name in batch_names])
		batch_y=[np.array([np.load('{}labels/{}{}{}{}'.format(self.pre_path,self.post_path,label_type,self.split,name[:-3] + 'npy'))
					for name in batch_names])
						for label_type in self.label_types]

		x1=tf.random.normal(shape=(1,4096))
		batch_y = [x1,batch_y[0], batch_y[1]]
		# print(batch_y[1].shape)
		
		for i in range(3) :
			if batch_y[i].shape[1]==1 :
				batch_y[i]=np.squeeze(batch_y[i],axis=1)

		channels=[256,65]#[desc,det]
		for i in range(1,3) :
			if batch_y[i].shape[1]==channels[i-1] :
				batch_y[i]=np.moveaxis(batch_y[i], 1,3)

		# correct output
		#return (batch_x,batch_y)
		
		#random values 
		# x = tf.random.normal((1,160,160,3))
		
		#x2=tf.random.normal(shape=(1,10,10,256))
		#x3=tf.random.normal(shape=(1,10, 10, 65))
		return (batch_x, batch_y)


if __name__=='__main__' :
	data=Dataset()
	print(data.__len__())
	data.__getitem__(0)
	#To be tested after generating labels successfully