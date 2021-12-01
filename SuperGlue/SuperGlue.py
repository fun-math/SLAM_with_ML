import torch

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

        self.kenc=KeypointEncoder(self.config['des_dim'],self.config['kpt_enc']) #Keypoint Encoder

        self.gnn=AttentionalGNN(self.config['des_dim'],self.config['GNN_layers']) #Attentional GNN

        self.final_proj=tf.keras.Sequential([
                      nn.Lambda(lambda x : tf.transpose(x,perm=[0,2,1]),input_shape=[self.config['des_dim'],None]),
                      nn.Conv1D(self.config['des_dim'], kernel_size=1),
                      nn.Lambda(lambda x : tf.transpose(x,perm=[0,2,1]))
                  ])   #Final Projection

        self.bin_score=tf.Variable([1.]) 

        ## Weight Loading code

    def call(self,data) :
        desc0, desc1 = data['desc0'], data['desc1']
        kpts0, kpts1 = data['kpts0'],data['kpts1']

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

        max0, max1 = tf.math.reduce_max(scores[:, :-1, :-1],2), tf.math.reduce_max(scores[:, :-1, :-1],1)
        indices0, indices1 = tf.math.argmax(scores[:, :-1, :-1],2), tf.math.argmax(scores[:, :-1, :-1],1)#max0.indices, max1.indices
        # print(indices1.shape,tf.gather(indices1,indices0,axis=1).shape)
        mutual0 = np.arange(indices0.shape[1]).reshape(1,-1) == gather(indices1,indices0,axis=1)
        mutual1 = np.arange(indices1.shape[1]).reshape(1,-1) == gather(indices0, indices1,axis=1)
        # zero = self.sparse_new_tensor(scores,0)
        zero=tf.constant(0,dtype=scores.dtype)
        # print(np.arange(indices0.shape[1]).reshape(1,-1).shape,gather(indices1,indices0,axis=1).shape)
        # print(mutual0.shape,max0.shape,zero.shape)
        mscores0 = tf.where(mutual0, tf.math.exp(max0), zero)
        mscores1 = tf.where(mutual1, gather(mscores0, indices1,1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_thresh'])
        valid1 = mutual1 & gather(valid0, indices1,1)
        indices0 = tf.where(valid0, indices0, tf.constant(-1,dtype=indices0.dtype))
        indices1 = tf.where(valid1, indices1, tf.constant(-1,dtype=indices1.dtype))

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def arange_like(self,x,dim):
        return tf.math.cumsum(tf.ones(x.shape[dim]))

def gather(params,indices,axis) :
    range=tf.repeat(tf.reshape(tf.range(indices.shape[0],dtype=tf.int64),(-1,1)),indices.shape[1],axis=1)
    idx=tf.stack([range,indices],axis=-1)
    return tf.gather_nd(params,idx)

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def postprocessing(scores) :
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    print(max0.values,indices0,max1.values,indices1)
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    # print(mutual0.shape,max0.values.exp().shape,zero)
    # print(arange_like(indices0, 1)[None],indices1.gather(1, indices0))
    # print(tf.gather(indices1,indices0,1))
    print(mutual0.shape)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > 0)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    return {
        'matches0': indices0, # use -1 for invalid match
        'matches1': indices1, # use -1 for invalid match
        'matching_scores0': mscores0,
        'matching_scores1': mscores1,
    }

if __name__=='__main__':
    superglue=SuperGlue()
    data={
        'kpts0' : tf.random.normal(shape=(2,5,2)),
        'desc0' : tf.random.normal(shape=(2,256,5)),
        'scores0' : tf.random.normal(shape=(2,5)),
        'shape0' : [320,240],#tf.constant([320,240]),

        'kpts1' : tf.random.normal(shape=(2,7,2)),
        'desc1' : tf.random.normal(shape=(2,256,7)),
        'scores1' : tf.random.normal(shape=(2,7)),
        'shape1' : [320,240],#tf.constant([320,240]),
    }
    # print(tf.constant([data['shape0'][0],data['shape0'][1]],shape=(1,2)))
    # print(data['shape0'])
    print(superglue(data))

    # scores=torch.rand(2,6,8)
    # postprocessing(scores)

    #checked

    