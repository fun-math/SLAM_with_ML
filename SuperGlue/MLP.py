import tensorflow as tf
import tensorflow.keras.layers as nn

__all__=["MLP","normalize_keypoints"]

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

def normalize_keypoints(kpts, image_shape):
    size=tf.constant([image_shape[1],image_shape[0]], shape=(1,2), dtype=kpts.dtype)
    center = size / 2
    scaling = tf.math.reduce_max(size, axis=1, keepdims=True) * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

if __name__=='__main__':
    channels=[32,32,64]
    model=MLP(channels)
    # model.build((None,10,2))
    # model.summary()
    print(model(tf.random.normal(shape=(1,10,2))).shape)
    x=tf.random.normal(shape=(2,5,2))
    # x=torch.rand(2,5,2)
    # xn=x.detach().numpy()
    print(normalize_keypoints(xn,[320,240]))#-torch_normalize_keypoints(x,[320,240]).detach().numpy())
    
