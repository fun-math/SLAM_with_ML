from SuperGlue import *
from Loss import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
# train_ds=Dataset(batch_size=16).tf_data()
# valid_ds=Dataset(split='val/',batch_size=16).tf_data()

#instantiate the model and run model.fit
model=SuperGlue()

# model.load_weights('/media/ironwolf/students/amit/SLAM_with_ML/weights/hfnet_new.h5')

model.custom_compile(tf.keras.optimizers.RMSprop(),Loss())

y=tf.random.uniform((15,3),0,2,tf.int32)
x={
    'kpts0' : tf.random.uniform(shape=(2,9,2)),
    'desc0' : tf.random.uniform(shape=(2,256,9)),
    'scores0' : tf.random.uniform((2,9),0,1),
    'shape0' : tf.constant([320,240],tf.float32,(1,2)),

    'kpts1' : tf.random.uniform(shape=(2,10,2)),
    'desc1' : tf.random.uniform(shape=(2,256,10)),
    'scores1' : tf.random.uniform((2,10),0,1),
    'shape1' : tf.constant([320,240],tf.float32,(1,2)),
}
# loss=Loss()
# print(loss(y_true,model(x)))
# print(model.train_step([x, y]))
#Checked
model.assign_data(train_ds=[(x,y),(x,y),(x,y)])
model.custom_fit(valid_freq=1000,step_init = 0)