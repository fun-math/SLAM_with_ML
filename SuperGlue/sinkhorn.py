import tensorflow as tf
class Sinkhorn():
    def __init__(self):
        self.iters = 3

    def sinkhorn(self, Z, log_mu, log_nu):
        u, v = tf.zeros_like(log_mu), tf.zeros_like(log_nu)
        # print(u.shape, v.shape)
        for _ in range(self.iters):
            u = log_mu - tf.math.log(tf.math.reduce_sum(tf.math.exp(Z + tf.expand_dims(v, 1)), axis=2))
            v = log_nu - tf.math.log(tf.math.reduce_sum(tf.math.exp(Z + tf.expand_dims(u, 2)), axis=1))
        return Z + tf.expand_dims(u, 2) + tf.expand_dims(v, 1)