import tensorflow as tf
import tensorflow.keras.layers as nn
import numpy as np
from copy import deepcopy

from KeypointEncoder import *
from AttentionalGNN import *
from sinkhorn import *

class SuperGlue(tf.keras.Model):
    def __init__(self, config=None):
        super(SuperGlue,self).__init__()

        self.config={
            'des_dim' : 256,
            'weights' : None,
            'kpt_enc' : [32,64,128,256],
            'GNN_layers' : ['self','cross']*9,
            'sinkhorn_iter' : 100,
            'match_thresh' : 0.2,
        }

        if config is not None :
            self.config=config

        self.kenc=None #Keypoint Encoder

        self.gnn=None #Attentional GNN

        self.final_proj=nn.Conv1D(self.config['des_dim'],1)

        self.bin_score=tf.Variable([1.])

        ## Weight Loading code

    def call(self,data) :
        desc0, desc1 = data['desc0'], data['desc1']
        kpts0, kpts1 = data['kpts0','kpts1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=tf.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=tf.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

            # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['shape0'])
        kpts1 = normalize_keypoints(kpts1, data['shape1'])

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = tf.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['des_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iter'])

        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = self.arange_like(indices0, 1)[None] == tf.gather(indices1,indices0,1)
        mutual1 = self.arange_like(indices1, 1)[None] == tf.gather(indices0, indices1,1)
        zero = self.sparse_new_tensor(scores,0)
        mscores0 = tf.where(mutual0, max0.values.exp(), zero)
        mscores1 = tf.where(mutual1, tf.gather(mscores0, indices1,1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_thresh'])
        valid1 = mutual1 & tf.gather(valid0, indices1,1)
        indices0 = tf.where(valid0, indices0, self.sparse_new_tensor(indices0,-1))
        indices1 = tf.where(valid1, indices1, self.sparse_new_tensor(indices1,-1))

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def sparse_new_tensor(self,x,v) :
        return tf.SparseTensor(x.indices,v*tf.ones_like(x.values),x.dense_shape)

    def arange_like(self,x,dim):
        tf.math.cumsum(tf.ones(x.shape[dim]))

        