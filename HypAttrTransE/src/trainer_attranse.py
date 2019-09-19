''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import sys

if '../../utils' not in sys.path:
    sys.path.append('../../utils')

import data  
import attranse


class Trainer(object):
    def __init__(self):
        self.batch_size=128
        self.dim=64
        self.this_data = None
        self.tf_parts = None
        self.save_path = 'this-attranse.ckpt'
        self.data_save_path = 'this-data.bin'
        self.L1=False

    def build(self, data, dim=64, batch_size=128, save_path = 'this-attranse.ckpt', data_save_path = 'this-data.bin', L1=False):
        self.this_data = data
        self.dim = self.this_data.dim = dim
        self.batch_size = self.this_data.batch_size = batch_size
        self.data_save_path = data_save_path
        self.save_path = save_path
        self.L1 = self.this_data.L1 = L1
        self.tf_parts = attranse.TFParts(num_rels=self.this_data.num_rels(),
                                 num_cons=self.this_data.num_cons(),
                                 dim=dim,
                                 batch_size=self.batch_size, L1=self.L1)

    def gen_A_batch(self, forever=False, shuffle=True):
        l = self.this_data.triples.shape[0]
        while True:
            triples = self.this_data.triples
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_size):
                batch = triples[i: i+self.batch_size, :]
                if batch.shape[0] < self.batch_size:
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_size
                neg_batch = self.this_data.corrupt_batch(batch)
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_h_batch, neg_t_batch = neg_batch[:, 0], neg_batch[:, 2]
                conf_batch = np.array([self.this_data.get_pp2c(h_batch[i], r_batch[i], t_batch[i], default = 0.) for i in range(len(h_batch))])
                neg_conf_batch = np.array([self.this_data.get_pp2c(neg_h_batch[i], r_batch[i], neg_t_batch[i], default = 0.) for i in range(len(neg_h_batch))])
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), neg_h_batch.astype(np.int64), neg_t_batch.astype(np.int64), conf_batch.astype(np.float32), neg_conf_batch.astype(np.float32)
            if not forever:
                break

    def train(self, epochs=20, save_every_epoch=10, lr=0.001, a1=0.1, m1=0.5, b=1., c=1., splitter='\t', line_end='\n'):
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.328)))
        sess.run(tf.initialize_all_variables())
        
        num_A_batch = len(list(self.gen_A_batch()))
        #num_C_batch = len(list(gen_C_batch(self.this_data, self.batch_size)))
        print('num_A_batch =', num_A_batch)
        
        # margins
        self.tf_parts._m1 = m1
        self.tf_parts._b = b
        self.tf_parts._c = c
        t0 = time.time()
        for epoch in range(epochs):
            epoch_loss = self.train1epoch(sess, num_A_batch, lr, a1, epoch + 1)
            print("Time use: %d" % (time.time() - t0))
            if np.isnan(epoch_loss):
                print("Training collapsed.")
                return
            if (epoch + 1) % save_every_epoch == 0:
                this_save_path = self.tf_parts._saver.save(sess, self.save_path)
                self.this_data.save(self.data_save_path)
                print("attranse saved in file: %s. Data saved in file: %s" % (this_save_path, self.data_save_path))
        this_save_path = self.tf_parts._saver.save(sess, self.save_path)
        self.this_data.save(self.data_save_path)
        print("attranse saved in file: %s" % this_save_path)
        sess.close()
        print("Done")

    def train1epoch(self, sess, num_A_batch, lr, a1, epoch):
        '''build and train a model.

        Args:
            self.batch_size: size of batch
            num_epoch: number of epoch. A epoch means a turn when all A/B_t/B_h/C are passed at least once.
            dim: dimension of embedding
            lr: learning rate
            self.this_data: a Data object holding data.
            save_every_epoch: save every this number of epochs.
            save_path: filepath to save the tensorflow model.
        '''

        this_gen_A_batch = self.gen_A_batch(forever=True)
        
        this_loss = []
        
        loss_A = loss_C_A = 0

        for batch_id in range(num_A_batch):
            # Optimize loss A
            A_h_index, A_r_index, A_t_index, A_hn_index, A_tn_index, C_batch, Cn_batch  = next(this_gen_A_batch)
            _, loss_A, _ = sess.run([self.tf_parts.trainstep, self.tf_parts._A_loss, self.tf_parts._clip_op1],
                    feed_dict={self.tf_parts._A_h_index: A_h_index, 
                               self.tf_parts._A_r_index: A_r_index,
                               self.tf_parts._A_t_index: A_t_index,
                               self.tf_parts._A_hn_index: A_hn_index,
                               self.tf_parts._A_tn_index: A_tn_index,
                               self.tf_parts._conf: C_batch,
                               self.tf_parts._conf_neg: Cn_batch,
                               self.tf_parts._lr: lr})
            loss_C_A = sess.run(self.tf_parts._C_loss_A,
                    feed_dict={self.tf_parts._A_h_index: A_h_index, 
                               self.tf_parts._A_r_index: A_r_index,
                               self.tf_parts._A_t_index: A_t_index})
            
            # Observe total loss
            batch_loss = [loss_A, loss_C_A]

            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            
            if ((batch_id + 1) % 50 == 0) or batch_id == num_A_batch - 1:
                print('process: %d / %d. Epoch %d' % (batch_id+1, num_A_batch, epoch))

        this_total_loss = np.sum(this_loss)
        print("Loss of epoch %d = %s" % (epoch, np.sum(this_total_loss)))
        print([l for l in this_loss])
        return this_total_loss

# A safer loading is available in Tester, with parameters like batch_size and dim recorded in the corresponding Data component
def load_tfparts(batch_size = 128,
                dim = 64,
                this_data=None,
                save_path = 'this-attranse.ckpt', L1=False):
    tf_parts = attranse.TFParts(num_rels=this_data.num_rels(),
                             num_cons=this_data.num_cons(),
                             dim=dim,
                             batch_size=self.batch_size, L1=L1)
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.328))) as sess:
        tf_parts._saver.restore(sess, save_path)