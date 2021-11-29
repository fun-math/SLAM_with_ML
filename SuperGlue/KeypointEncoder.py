import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np

from MLP import *

class KeypointEncoder(tf.keras.layers.Layer) :
    def __init__(self, feature_dim, channels) :
        super(KeypointEncoder, self).__init__()

        self.encoder = MLP(channels+[feature_dim])

    def call(self, kpts, scores):
        # inputs=tf.concat([tf.transpose(kpts,perm=[0,2,1]),scores],axis=1)
        inputs=tf.concat([tf.transpose(kpts,perm=[0,2,1]),
                        tf.keras.backend.expand_dims(scores,1)],axis=1)
        return self.encoder(inputs)

if __name__=='__main__':
    model=KeypointEncoder(256,[32,32])
    # model.build((None,10,3))
    print(model(tf.random.normal(shape=(2,5,2)),
                tf.random.normal(shape=(2,5))).shape)
    # model.summary()
    #checked
    