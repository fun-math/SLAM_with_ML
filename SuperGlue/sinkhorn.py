import tensorflow as tf
import torch

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    u, v = tf.zeros_like(log_mu), tf.zeros_like(log_nu)
    
    for _ in range(iters):
        u = log_mu - tf.math.reduce_logsumexp(Z + tf.expand_dims(v, 1), axis=2)
        v = log_nu - tf.math.reduce_logsumexp(Z + tf.expand_dims(u, 2), axis=1)
    return Z + tf.expand_dims(u, 2) + tf.expand_dims(v, 1)

def log_optimal_transport(scores, alpha, iters: int):
    b, m, n = scores.shape
    _, ms, ns = 0,m*1.0,n*1.0

    bins0 = alpha*tf.ones((b, m, 1))
    bins1 = alpha*tf.ones((b, 1, n))
    alpha = alpha*tf.ones((b, 1, 1))

    couplings = tf.concat([tf.concat([scores, bins0], -1),
                           tf.concat([bins1, alpha], -1)], 1)

    norm = - tf.reshape(tf.math.log(ms + ns),shape=(1,1))
    
    log_mu = tf.concat([norm*tf.ones((m,1)), tf.math.log(ns)[None] + norm],axis=0)
    log_nu = tf.concat([norm*tf.ones((n,1)), tf.math.log(ms)[None] + norm],axis=0)
    log_mu, log_nu = tf.repeat(log_mu[None],b,axis=0), tf.repeat(log_nu[None],b,axis=0)
    
    Z = log_sinkhorn_iterations(couplings, tf.reshape(log_mu,shape=(b,m+1)), tf.reshape(log_nu,shape=(b,n+1)), iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

####torch code
def t_log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def t_log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = t_log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z
####torch code

if __name__=='__main__':
    # scores=tf.random.normal(shape=(2,5,4))
    # thresh=tf.random.normal(shape=(1,1))
    scores=torch.rand(2,5,4)
    thresh=torch.rand(1)
    print(t_log_optimal_transport(scores,thresh,3).numpy())
    #checked