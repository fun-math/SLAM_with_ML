from dataset import *
from HFnet import *

#Instantiate dataset class and convert it into tf.dataset instance 
train_ds=Dataset(batch_size=2)
valid_ds=Dataset(batch_size=2)

#instantiate the model and run model.fit
model=HFnet(in_shape = (160, 160, 3))
model.compile(optimizer = "RMSprop", 
        loss = [Loss_desc('gdesc'),Loss_desc('ldesc'),Loss_ldet()])

model.assign_data(train_ds=train_ds,valid_ds=valid_ds)
model.custom_fit(8,4)