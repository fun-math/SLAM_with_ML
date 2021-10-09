import tensorflow as tf
import numpy as np
from copy import deepcopy

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads: int, d_model: int):
    super(MultiHeadAttention,self).__init__()  #Use the hfnet 
    if d_model%num_heads==0:
      self.dim = d_model//num_heads
      self.num_heads = num_heads
      self.merge = tf.keras.layers.Conv1d(d_model, kernel_size=1)
      self.proj = [deepcopy(self.merge) for _ in range(3)] ## torch is torch.nn.Modulelist which is a list of submodules so I have used concatenate

  def call(self, query, key, value):
      batch_dim = query.size[0]
      query, key, value = [l(x).reshape(batch_dim, self.dim, self.num_heads, -1) for l, x in zip(self.proj, (query, key, value))]
      x, _ = attention(query, key, value)

      return self.merge()
    
def attention(query, key, value):
  dim = query.shape[1]
  scores = tf.einsum('bdhn, bdhm->bdhn', query, key)/np.sqrt(dim)
  prob = tf.nn.softmax(scores, axis=-1) 
  return tf.einsum('bhnm, bdhm->bdhn', prob, value), prob
