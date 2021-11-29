import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as nn
from copy import deepcopy

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model):
    super(MultiHeadAttention,self).__init__()  #Use the hfnet 
    if d_model%num_heads==0:
      self.dim = d_model//num_heads
      self.num_heads = num_heads
      self.merge = tf.keras.Sequential([
                      nn.Lambda(lambda x : tf.transpose(x,perm=[0,2,1])),
                      nn.Conv1D(d_model, kernel_size=1),
                      nn.Lambda(lambda x : tf.transpose(x,perm=[0,2,1]))
                  ])
      self.proj = [deepcopy(self.merge) for _ in range(3)] ## torch is torch.nn.Modulelist which is a list of submodules so I have used concatenate

  def call(self, query, key, value):
      batch_dim = query.shape[0]
      query, key, value = [tf.reshape(l(x),(batch_dim, self.dim, self.num_heads, -1)) for l, x in zip(self.proj, (query, key, value))]
      x, _ = attention(query, key, value)

      return self.merge(tf.reshape(x,(batch_dim,self.dim*self.num_heads,-1)))#contiguous
    
def attention(query, key, value):
  dim = int(query.shape[1])
  scores = tf.einsum('bdhn, bdhm->bhnm', query, key)/np.sqrt(dim)
  prob = tf.nn.softmax(scores, axis=-1) 
  return tf.einsum('bhnm, bdhm->bdhn', prob, value), prob

if __name__=='__main__':
  layer=MultiHeadAttention(4,256)
  x=tf.random.normal(shape=(2,256,5))
  y=tf.random.normal(shape=(2,256,6))
  a=layer(x,y,y)
  print(a.shape)
  #checked

