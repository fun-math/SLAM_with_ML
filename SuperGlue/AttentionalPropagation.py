import tensorflow as tf

from MultiHeadAttention import *
from MLP import *

class AttentionalPropagation(tf.keras.layers.Layer):
  def __init__(self, feature_dim, num_heads):
    super(AttentionalPropagation,self).__init__()
    self.attention = MultiHeadAttention(num_heads, feature_dim)
    self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
    # tf.zeros_like(self.mlp[-1].bias) Set bias to zero

  def call(self, x, source):
    msg = self.attention(x, source, source)
    return self.mlp(tf.concat([x,msg], axis=1))
    
if __name__=='__main__':
  layer=AttentionalPropagation(256,4)
  x=tf.random.normal(shape=(2,256,4))
  y=tf.random.normal(shape=(2,256,5))
  print(layer(x,y).shape)

