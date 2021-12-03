import tensorflow as tf

class Loss(tf.keras.losses.Loss) :
    def __init__(self) :
	    super(Loss,self).__init__()
    
    def call(self,y_true,y_pred) :
        '''
        y_true : A 2-D tensor of shape (None, 2)

        y_true contains for each pair of image in the batch,
        1) (a,b) if key point number a of first image and key point number b of second image are matched
        2) (i,N+1) if key point number i of first image is unmatched
        3) (M+1,j) if key point number j of second image is unmatched

        Thus, the first dimension of y_true should be sum over all pairs in the batch of number of matched points and unmatched points

        PS : M,N is the no. of key points in the first, second image
        '''

        return -tf.math.reduce_mean(tf.math.log(tf.gather_nd(y_pred,y_true)))
        #Accordig to the paper, it should be reduce_sum(...)/batch_size instead of reduce_mean(...)

if __name__=='__main__' :
    y_pred=tf.random.uniform((9,10),0,1)
    y_true=tf.random.uniform((15,2),0,9,tf.int32)
    loss=Loss()
    # print(tf.gather_nd(y_pred,y_true))
    print(loss.call(y_true,y_pred).numpy())
    