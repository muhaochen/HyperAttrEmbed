from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if '../src' not in sys.path:
    sys.path.append('../src')

import os

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time

if '../../utils' not in sys.path:
    sys.path.append('../../utils')

import data  
import transe
import trainer_transe
from tester_transe import Tester

model_path = './enron-transe.ckpt'
data_path = './enron-data.bin'
test_data = ["../../../enron/gender_identified_enron_corpus/graph/partitions/structure_employee_wemail_random_test0.2.csv"]
vocab_data = ["../../../enron/gender_identified_enron_corpus/graph/structure_employee_wemail.csv"]
result_file = './test_enron_transe_emp_emailgraph.txt'

if len(sys.argv) > 1:
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    test_data = [sys.argv[3]]
    vocab_data = [sys.argv[4]]
    result_file = sys.argv[5]

TopK = 10

tester = Tester()
tester.build(save_path = model_path, data_save_path = data_path)
tester.load_test_data_enron_csv(filenames=test_data)

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()

index = Value('i', 0, lock=True) #index
score_head = manager.list() #scores for each case (and its rel)
rank_head = manager.list() #rank, and its rel

vocab = tester.read_vocab_enron_csv(filenames=vocab_data)


t0 = time.time()

def test(tester, index, score_head, rank_head):
    while index.value < len(tester.test_triples):
        id = index.value
        index.value += 1
        if id % 100 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
        h, r, t = tester.test_triples[id]
        head_pool = tester.vec_c
        head_vec = tester.vec_c[h]
        tail_vec = tester.vec_c[t]
        rel_vec = tester.rel_index2vec(r)
        #Change
        target_vec = tail_vec - rel_vec
        this_rank_head = tester.rank_index_from(target_vec, head_pool, h, cand_scope = vocab, self_id = h) * 1.
        this_score_head = np.zeros(TopK)
        hit = False
        pred = tester.kNN(target_vec, head_pool, topk = TopK, cand_scope = vocab, self_id=None)
        for i in range(len(pred)):
            if not hit and pred[i][0] in tester.tr_map[t][r]:
                hit = True
            if hit:
                this_score_head[i] = 1.
        rank_head.append(this_rank_head)
        score_head.append(this_score_head)

print('Average norm=%d' % np.mean([LA.norm(vec, ord=(1 if tester.this_data.L1 else 2)) for vec in tester.vec_c]))
processes = [Process(target=test, args=(tester, index, score_head, rank_head)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

with open(result_file, 'w') as fp:
    #head
    fp.write('Head\n')
    mean_rank = 0.0
    mrr = 0.0
    hits = np.zeros(TopK)
    for s in rank_head:
        mean_rank += s
        mrr += 1. / s
    mean_rank /= len(rank_head)
    fp.write('rank='+str(mean_rank) + '\n')
    mrr /= len(rank_head)
    fp.write('over all='+str(mrr) + '\n')
    for s in score_head:
        hits += s
    hits /= len(score_head)
    fp.write('overall='+'\t'.join(str(x) for x in hits) + '\n')
    #trsym