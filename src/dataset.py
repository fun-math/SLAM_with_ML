import tensorflow as tf
import numpy as np
import cv2
import glob

#/media/ironwolf/students/amit/datasets/bdd100k/images/100k/train/
#/media/ironwolf/students/amit/datasets/bdd100k/labels/100k/<label_type>/train/
# <label_type> is one of ['gdesc/','ldesc/','ldet/']

class Dataset(tf.keras.utils.Sequence) :
	def __init__(self,pre_path="/media/ironwolf/students/amit/datasets/bdd100k/",
		post_path='100k/',split='train/',batch_size=64,cores=8) :
		
		self.pre_path=pre_path
		self.post_path=post_path
		self.split=split
		self.batch_size=batch_size
		self.cores=cores
		self.label_types=[ 'gdesc/', 'ldesc/','ldet/']
		self.names=[full_name.split('/')[-1] 
			for full_name in glob.glob(pre_path+'images/'+post_path+split+'*')]

	def __len__(self) :
		#random value
		# return 4

		#correct output
		return np.ceil(len(self.names)/self.batch_size)

	def _imread(self,path) :
		
		#print(path)
		img=cv2.imread(path)
		img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = img.astype('float')/255.0
		img = np.expand_dims(img, axis = -1) 
		img = np.repeat(img, 3, axis = -1).astype(np.float32)
		#print(img.shape)
		#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		return img

	def __getitem__(self,idx) :
		# batch_names=self.names[idx*self.batch_size : (idx+1)*self.batch_size]

		# batch_x=np.array([self._imread('{}images/{}{}{}'.format(self.pre_path,self.post_path,self.split,name))
		# 		for name in batch_names])
		# batch_y=[np.array([np.load('{}labels/{}{}{}{}'.format(self.pre_path,self.post_path,label_type,self.split,name[:-3] + 'npy'))
		# 			for name in batch_names])
		# 				for label_type in self.label_types]

		
		# for i in range(3) :
		# 	if batch_y[i].shape[1]==1 :
		# 		batch_y[i]=np.squeeze(batch_y[i],axis=1)

		# channels=[256,65]#[desc,det]
		# for i in range(1,3) :
		# 	if batch_y[i].shape[1]==channels[i-1] :
		# 		batch_y[i]=np.moveaxis(batch_y[i], 1,3)

		# correct output
		#return (batch_x,batch_y)
		
		#random values 
		x = tf.random.normal((self.batch_size,160,160,3))
		x1=tf.random.normal((self.batch_size,4096))
		x2=tf.random.normal(shape=(self.batch_size,10,10,256))
		x3=tf.random.normal(shape=(self.batch_size,10, 10, 65))
		return (x1, [x1,x2,x3])

	def parse_function(self,name) :
		image = tf.io.read_file('{}images/{}{}{}'.format(self.pre_path,self.post_path,self.split,name))
		image = tf.image.decode_jpeg(image, channels=3)
		image = tf.image.resize(image, [480, 640])
		image = tf.image.convert_image_dtype(image, tf.float32)/255.0

		y1,y2,y3=[np.load('{}labels/{}{}{}{}'.format(
			self.pre_path,self.post_path,label_type,self.split,name[:-3] + 'npy')
		)[0].astype(np.float32) for label_type in self.label_types]

		y2,y3 = [np.moveaxis(y, 0,2) for y in [y2,y3]]
		return image,y1,y2,y3

	def tf_data(self) :
		return tf.data.Dataset.from_tensor_slices(self.names
			).shuffle(len(self.names)
			).map(lambda name : tf.numpy_function(self.parse_function,[name],
				4*[tf.float32]),num_parallel_calls=self.cores
			).batch(self.batch_size
			).prefetch(4)
		'''
		data=tf.data.Dataset.from_generator(
			lambda : (s for s in self),
			output_signature=(tf.TensorSpec((self.batch_size,160,160,3),tf.float32),
				[tf.TensorSpec((self.batch_size,4096),tf.float32),
				tf.TensorSpec((self.batch_size,10,10,256),tf.float32),
				tf.TensorSpec((self.batch_size,10,10,65),tf.float32)],
				))

		return data
		'''



if __name__=='__main__' :
	data=Dataset()
	print(data.__len__())
	data.__getitem__(0)
	data=data.tf_data()
	for x,y1,y2,y3 in data :
		print(x.shape,y1.shape,y2.shape,y3.shape)
	#To be tested after generating labels successfully
