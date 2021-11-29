import tensorflow as tf
from AttentionalPropagation import *

class AttentionalGNN(tf.keras.layers.Layer):
  def __init__(self, feature_dim, layer_names):
    super(AttentionalGNN, self).__init__()
    self.layers = [AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))]
    self.names = layer_names

  def call(self, desc0, desc1):
    for layer, name in zip(self.layers, self.names):
      if name=='cross':
        src0, src1 = desc1, desc0
      else:
        src0, src1 = desc0, desc1
      delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
      desc0, desc1 = (desc0  + delta0), (desc1 + delta1)
    return desc0, desc1

if __name__=='__main__':
  gnn=AttentionalGNN(256,['self','cross']*3)
  print(gnn(tf.random.normal(shape=(2,256,4)),tf.random.normal(shape=(2,256,4)))[0].shape)