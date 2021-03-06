'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sys
if '../../utils' not in sys.path:
    sys.path.append('../../utils')

import data as pymod_data
from data import Data
import pickle

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer

class TFParts(object):
    '''TensorFlow-related things. 
    
    This is to keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels, num_cons, dim, batch_size, L1=False):
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology. 
        self._batch_size = batch_size
        self._epoch_loss = 0
        # margins
        self._m1 = 0.25
        self._b = 1.
        self._c = 1.
        self.L1 = L1
        self.eps = 1e-6
        self.build()
     
    def poincare_dist(self, u, v):
        uu, uv, vv = tf.norm(u, axis=-1)**2, tf.norm(u-v, axis=-1)**2, tf.norm(v, axis=-1)**2
        alpha, beta = tf.maximum(1.-uu,self.eps), tf.maximum(1.-vv,self.eps)
        gamma = tf.maximum(1.+2.*uv/alpha/beta,1+self.eps)
        return tf.acosh(gamma)
    
    def proj(self,x):
        return tf.clip_by_norm(x,1.-self.eps,axes=-1)

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size    

    def build(self):
        tf.reset_default_graph()

        with tf.variable_scope("graph", initializer=orthogonal_initializer()):
            # Variables (matrix of embeddings/transformations)

            self._hto = hto = tf.get_variable(
                name='ht',  # for t AND h
                shape=[self.num_cons, self.dim],
                dtype=tf.float32)
            self._ro = ro = tf.get_variable(
                name='r',
                shape=[self.num_rels, self.dim],
                dtype=tf.float32)
            # disable l2_normalize if use regularization
            self._ht = ht = hto
            self._r = r = ro

            self._ht_assign = ht_assign = tf.placeholder(
                name='ht_assign',
                shape=[self.num_cons, self.dim],
                dtype=tf.float32)
            self._r_assign = r_assign = tf.placeholder(
                name='r_assign',
                shape=[self.num_rels, self.dim],
                dtype=tf.float32)

            # Type A loss : [|| h + r - t ||_2 + m1 - || h + r - t ||_2]+    here [.]+ means max (. , 0)
            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_t_index')
            self._A_hn_index = A_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_hn_index')
            self._A_tn_index = A_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_tn_index')
            
            self._conf = Conf = tf.placeholder(
                dtype=tf.float32,
                shape=[self.batch_size],
                name='Conf')
            self._conf_neg = Conf_neg = tf.placeholder(
                dtype=tf.float32,
                shape=[self.batch_size],
                name='Conf_neg')

            A_h_con_batch = tf.nn.embedding_lookup(ht, A_h_index)
            A_t_con_batch = tf.nn.embedding_lookup(ht, A_t_index)
            A_rel_batch = tf.nn.embedding_lookup(r, A_r_index)
           
            A_hn_con_batch = tf.nn.embedding_lookup(ht,A_hn_index)
            A_tn_con_batch = tf.nn.embedding_lookup(ht,A_tn_index)
            
            # This stores h  + r - t 
            A_loss_matrix = tf.subtract(tf.add(A_h_con_batch, A_rel_batch), A_t_con_batch)
            # This stores h' + r - t' for negative samples
            A_neg_matrix = tf.subtract(tf.add(A_hn_con_batch, A_rel_batch), A_tn_con_batch)
            # L-2 norm
            # [||h M_hr + r - t M_tr|| + m1 - ||h' M_hr + r - t' M_tr||)]+     here [.]+ means max (. , 0)
            if self.L1:
                self._A_loss = A_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.subtract( tf.add( tf.multiply(tf.add(Conf, self._b), tf.reduce_sum(tf.abs(A_loss_matrix), 1) ), self._m1 * self._b),
                    tf.multiply( tf.add(Conf_neg / self._c, self._b), tf.reduce_sum(tf.abs(A_neg_matrix), 1) )), 
                    0.)
                )  /  self.batch_size
            else:
                self._A_loss = A_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.subtract(tf.add( tf.multiply( tf.add(Conf / self._c, self._b), tf.sqrt(tf.reduce_sum(tf.square(A_loss_matrix), 1)  )), self._m1  * self._b),
                    tf.multiply( tf.add(Conf_neg, self._b), tf.sqrt(tf.reduce_sum(tf.square(A_neg_matrix), 1)) ) ), 
                    0.)
                )  /  self.batch_size

            A_vec_restraint = tf.concat([tf.abs(tf.subtract(tf.sqrt(tf.reduce_sum(tf.square(A_h_con_batch), 1)), 1.)), tf.abs(tf.subtract(tf.sqrt(tf.reduce_sum(tf.square(A_t_con_batch), 1)), 1.))], 0)

            A_rel_restraint = tf.maximum(tf.subtract(tf.sqrt(tf.reduce_sum(tf.square(A_rel_batch), 1)), 2.), 0.)


            # Type C loss : Soft-constraint on vector norms
            self._C_loss_A = C_loss_A = tf.reduce_sum(A_vec_restraint)

            # Force normalize pre-projected vecs
            
            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            self._opt = opt = tf.train.AdamOptimizer(lr)#AdagradOptimizer(lr)#tf.train.AdamOptimizer(lr)#GradientDescentOptimizer(lr)
            grad_vars = opt.compute_gradients(A_loss)
            #print (grad_vars.shape)
            rescaled = [(g*(1.-tf.reshape(tf.norm(v,axis=1),(-1,1))**2)**2/4.,v) for g,v in grad_vars] #?
            self.trainstep = opt.apply_gradients(rescaled)
               
            self._train_op_A = train_op_A = opt.minimize(A_loss)
            self._train_op_C_A = train_op_C_A = opt.minimize(C_loss_A)

            self._assign_ht_op = assign_ht_op = hto.assign(ht_assign)
            self._assign_r_op = assign_r_op = self._ro.assign(r_assign)
            self._clip_op1 = tf.assign(ht, self.proj(ht))
            self._clip_op2 = tf.assign(r, self.proj(r))
            # Saver
            self._saver = tf.train.Saver()