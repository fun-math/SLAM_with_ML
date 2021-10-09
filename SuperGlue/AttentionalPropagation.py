import tensorflow as tf

from MultiHeadAttention import *
from MLP import *

class AttentionalPropagation(tf.keras.layers.Layer):
  def __init__(self, feature_dim, num_heads):
    super(AttentionalPropagation,self).__init__()
    self.attention = MultiHeadAttention(num_heads, feature_dim)
    self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
    tf.zeros_like(self.mlp[-1].bias)

  def call(self, x, source):
    msg = self.attention(x, source, source)
    return self.mlp(tf.concat([msg, x], dim=1))
    