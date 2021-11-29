import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np

__all__=["MLP"]

def MLP(channels, bn=True) :
    '''
    channels 
    '''
    model = tf.keras.Sequential([])

    model.add(nn.Lambda(lambda x : tf.transpose(x,perm=[0,2,1])))
    for i in range(len(channels)) :
        model.add(nn.Conv1D(channels[i],1))
        if bn :
            model.add(nn.BatchNormalization())
        model.add(nn.ReLU())
    model.add(nn.Lambda(lambda x : tf.transpose(x,perm=[0,2,1])))

    return model

if __name__=='__main__':
    channels=[32,32,64]
    model=MLP(channels)
    # model.build((None,10,2))
    # model.summary()
    print(model(tf.random.normal(shape=(1,10,2))).shape)
    
