from dataset import *
from HFnet import *
import tensorflow as tf
print('*************************************    GPU device   **********************************', tf.config.list_physical_devices('GPU'))

# tf.debugging.set_log_device_placement(True)
# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# with strategy.scope():
if True :
    for line in open('/media/ironwolf/students/amit/SLAM_with_ML/src/steps.txt'):
        if line.strip():
            n = int(line)
    print('********************', n)
    
    train_ds=Dataset(batch_size=4)
    valid_ds=Dataset(batch_size=4)

#instantiate the model and run model.fit
    model=HFnet()
    #model.load_weights('/media/ironwolf/students/amit/SLAM_with_ML/weights/hfnet_last.h5')
# model.compile(optimizer = "RMSprop", 
        # loss = [Loss_desc('gdesc'),Loss_desc('ldesc'),Loss_ldet()])
    model.assign_loss([Loss_desc('gdesc'),Loss_desc('ldesc'),Loss_ldet()])
    model.assign_data(train_ds=train_ds,valid_ds=valid_ds)
    # print('**************************************************************************************', tf.__version__)
    model.custom_fit(valid_freq=1000, step_init = n)