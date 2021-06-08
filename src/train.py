
from dataset import *
from HFnet import *
import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras import mixed_precision
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# mixed_precision.experimental.set_policy('mixed_float16')
# print('*************************************    GPU device   **********************************', tf.config.list_physical_devices('GPU'))
# tf.enable_eager_execution()
# tf.debugging.set_log_device_placement(True)
# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# with strategy.scope():
if True :
    for line in open('/media/ironwolf/students/amit/SLAM_with_ML/src/steps.txt'):
        if line.strip():
            n = int(line)
    print('********************', n)
    
    train_ds=Dataset(batch_size=16)
    valid_ds=Dataset(split='val/',batch_size=16)

#instantiate the model and run model.fit
    model=HFnet(in_shape=(480,640,3))
    

    # import pdb; pdb.set_trace()
    model.build(input_shape=(None, 480, 640, 3))
    model.assign_data(train_ds=train_ds,valid_ds=valid_ds)
    model.load_weights('/media/ironwolf/students/amit/SLAM_with_ML/weights/hfnet_new.h5')
    
# model.compile(optimizer = "RMSprop", 
        # loss = [Loss_desc('gdesc'),Loss_desc('ldesc'),Loss_ldet()])
    model.assign_loss([Loss_desc('gdesc'),Loss_desc('ldesc'),Loss_ldet()])
    # img = cv2.imread('/media/ironwolf/students/amit/datasets/bdd100k/images/100k/test/cabc30fc-e7726578.jpg')
    # img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = img.astype('float')/255.0
    # img = np.expand_dims(img, axis = -1) 
    # img = np.repeat(img, 3, axis = -1).astype(np.float32)
    # img = [img]




    # a = model(img)
    # print(a.shape)

    model.custom_fit(valid_freq=1000,step_init = n)