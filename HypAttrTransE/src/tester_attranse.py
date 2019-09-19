''' Module for held-out test.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import heapq as HP

import sys
if '../../utils' not in sys.path:
    sys.path.append('../../utils')

import data  
import attranse
import trainer_attranse
import csv

# This class is used to load and combine a TF_Parts and a Data object, and provides some useful methods for training
class Tester(object):
    def __init__(self):
        self.tf_parts = None
        self.this_data = None
        self.vec_c = np.array([0])
        self.vec_r = np.array([0])
        # below for test data
        self.test_triples = np.array([0])
        self.test_triples_group = {}
        # hr2t
        self.hr_map = {}
        # tr2h
        self.tr_map = {}
        self.ht_map = {}
        # for span trees
        self.span = []
        self.e2span = {}
    
    def build(self, save_path = 'this-model.ckpt', data_save_path = 'this-data.bin'):
        self.this_data = data.Data()
        self.this_data.load(data_save_path)
        self.tf_parts = attranse.TFParts(num_rels=self.this_data.num_rels(),
                         num_cons=self.this_data.num_cons(),
                         dim=self.this_data.dim,
                         batch_size=self.this_data.batch_size, L1=self.this_data.L1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.328)))
        self.tf_parts._saver.restore(sess, save_path)  # load it
        value_ht, value_r = sess.run([self.tf_parts._ht, self.tf_parts._r])  # extract values.
        sess.close()
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        # when a model doesn't have Mt, suppose it should pass Mh

    def load_test_data(self, filename, splitter = '\t', line_end = '\n'):
        num_lines = 0
        triples = []
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) < 3:
                continue
            num_lines += 1
            h = self.this_data.con_str2index(line[0])
            r = self.this_data.rel_str2index(line[1])
            t = self.this_data.con_str2index(line[2])
            if h is None or r is None or t is None:
                continue
            triples.append([h, r, t])
            if self.hr_map.get(h) is None:
                self.hr_map[h] = {}
            if self.hr_map[h].get(r) is None:
                self.hr_map[h][r] = set([t])
            else:
                self.hr_map[h][r].add(t)
            if self.tr_map.get(t) is None:
                self.tr_map[t] = {}
            if self.tr_map[t].get(r) is None:
                self.tr_map[t][r] = set([h])
            else:
                self.tr_map[t][r].add(h)
            if self.ht_map.get(h) is None:
                self.ht_map[h] = {}
            if self.ht_map[h].get(t) is None:
                self.ht_map[h][t] = set([r])
            else:
                self.ht_map[h][t].add(r)
            # add to group
            if self.test_triples_group.get(r) is None:
                self.test_triples_group[r] = [(h, r, t)]
            else:
                self.test_triples_group[r].append((h, r, t))
        self.test_triples = np.array(triples)
        print("Loaded test data from %s, %d out of %d." % (filename, len(triples), num_lines))

    def get_span(self, eid):
        sid = self.e2span.get(eid)
        if sid is None:
            return None
        return self.span[sid]

    def load_test_data_enron_csv(self, filenames=[], rel_sp=['supervises']):
        num_lines = 0
        triples = []
        last_span = -1
        nullify = 1
        assert(len(filenames) > 0)
        for ifile in filenames:
            for line in csv.reader(open(ifile), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL):
                if len(line) < 5:
                    continue
                num_lines += 1
                h = self.this_data.con_str2index(line[0])
                r = self.this_data.rel_str2index(line[2])
                t = self.this_data.con_str2index(line[3])
                if h is None or r is None or t is None:
                    continue
                triples.append([h, r, t])
                if self.hr_map.get(h) is None:
                    self.hr_map[h] = {}
                if self.hr_map[h].get(r) is None:
                    self.hr_map[h][r] = set([t])
                else:
                    self.hr_map[h][r].add(t)
                if self.tr_map.get(t) is None:
                    self.tr_map[t] = {}
                if self.tr_map[t].get(r) is None:
                    self.tr_map[t][r] = set([h])
                else:
                    self.tr_map[t][r].add(h)
                if self.ht_map.get(h) is None:
                    self.ht_map[h] = {}
                if self.ht_map[h].get(t) is None:
                    self.ht_map[h][t] = set([r])
                else:
                    self.ht_map[h][t].add(r)
                # add to group
                if self.test_triples_group.get(r) is None:
                    self.test_triples_group[r] = [(h, r, t)]
                else:
                    self.test_triples_group[r].append((h, r, t))
                # store span components
                if line[2] in rel_sp:
                    if self.e2span.get(h) is None and self.e2span.get(t) is None:
                        last_span += 1
                        self.span.append(set([h,t]))
                        self.e2span[h] = self.e2span[t] = last_span
                    elif self.e2span.get(h) is None:
                        self.span[self.e2span[t]].add(h)
                        self.e2span[h] = self.e2span[t]
                    elif self.e2span.get(t) is None:
                        self.span[self.e2span[h]].add(t)
                        self.e2span[t] = self.e2span[h]
                    else:
                        if self.e2span[h] != self.e2span[t]:
                            self.span[self.e2span[h]] = self.span[self.e2span[h]] | self.span[self.e2span[t]]
                            temp = self.span[self.e2span[t]]
                            self.span[self.e2span[t]] = None
                            for x in temp:
                                if x!=h:
                                    self.e2span[x] = self.e2span[h]
                            self.e2span[t] = self.e2span[h]
                            nullify += 1
        self.test_triples = np.array(triples)
        print("Loaded test data %d out of %d." % (len(triples), num_lines))
        print("Spans",last_span + 1 - nullify)
                
    def read_test_data_enron_csv(self, filenames=[]):
        return self.this_data.read_triples_enron_csv(filenames=filenames)

    def read_vocab_enron_csv(self, filenames = []):
        return self.this_data.read_vocab_enron_csv(filenames =filenames )
    
    def con_index2vec(self, c):
        return self.vec_c[c]

    def rel_index2vec(self, r):
        return self.vec_r[r]

    def con_str2vec(self, str):
        this_index = self.this_data.con_str2index(str)
        if this_index is None:
            return None
        return self.vec_c[this_index]
    
    def rel_str2vec(self, str):
        this_index = self.this_data.rel_str2index(str)
        if this_index is None:
            return None
        return self.vec_r[this_index]
    
    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return
        def __lt__(self, other):
            return self.dist > other.dist
                
    def con_index2str(self, str):
        return self.this_data.con_index2str(str)
    
    def rel_index2str(self, str):
        return self.this_data.rel_index2str(str)
    
    # input must contain a pool of pre- or post-projected vecs. return a list of indices and dist
    def kNN(self, vec, vec_pool, cand_scope=None, topk=10, self_id=None):
        q = []
        if cand_scope is None:
            cand_scope = range(len(vec_pool))
        for i in cand_scope:
            #skip self
            if i == self_id:
                continue
            dist = self.dist(vec, vec_pool[i])
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, self.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst
        
    # input must contain a pool of pre- or post-projected vecs. return a list of indices and dist. rank an index in a vec_pool from 
    def rank_index_from(self, vec, vec_pool, index, cand_scope=None, self_id = None):
        dist = self.dist(vec, vec_pool[index])
        rank = 1
        if cand_scope is None:
            cand_scope = range(len(vec_pool))
        for i in cand_scope:
            if i == index or i == self_id:
                continue
            if dist > self.dist(vec, vec_pool[i]):
                rank += 1
        return rank

    def dist(self, v1, v2):
        return self.tf_parts.poincare_dist(v1, v2)

    def dissimilarity(self, h, r, t):
        h_vec = self.vec_c[h]
        t_vec = self.vec_c[t]
        return self.tf_parts.poincare_dist(h_vec + self.vec_r[r], t_vec)