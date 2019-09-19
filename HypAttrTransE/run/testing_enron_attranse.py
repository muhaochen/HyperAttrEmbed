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
import attranse
import trainer_attranse
from tester_attranse import Tester

model_path = './enron-attranse.ckpt'
data_path = './enron-data.bin'
test_data = ["../../../enron/gender_identified_enron_corpus/graph/partitions/structure_employee_wemail_random_test0.2.csv"]
vocab_data = ["../../../enron/gender_identified_enron_corpus/graph/structure_employee_wemail.csv"]
result_file = './test_enron_attranse_emp_emailgraph.txt'

if len(sys.argv) > 1:
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    test_data = [sys.argv[3]]
    vocab_data = [sys.argv[4]]
    result_file = sys.argv[5]

TopK = 10

tester = Tester()
tester.build(save_path = model_path, data_save_path = data_path)
tester.load_test_data_enron_csv(filenames=test_data, rel_sp='supervises')

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()

rank_list = manager.list()
prec_list = manager.list()

index = Value('i', 0, lock=True) #index
total = Value('i', 0, lock=True) #index

vocab = tester.read_vocab_enron_csv(filenames=vocab_data)


t0 = time.time()

def test(tester, index, rank_list, prec_list):
    while index.value < len(tester.test_triples):
        id = index.value
        index.value += 1
        if id % 100 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
        h, r, t = tester.test_triples[id]
        span = tester.get_span(t)
        if span is None:
            continue
        head_pool = tester.vec_c
        head_vec = tester.vec_c[h]
        tail_vec = tester.vec_c[t]
        rel_vec = tester.rel_index2vec(r)
        #Change
        target_vec = tail_vec - rel_vec
        this_rank_list = np.zeros(TopK)
        hit = False
        pred = tester.kNN(target_vec, head_pool, topk = len(span), cand_scope = span, self_id=None)
        prec_list.append((pred[0][0], r, t))
        for x in pred:
            rank_list.append((x[0], r, t, x[1]))
        total.value += 1

print('Average norm=%d' % np.mean([LA.norm(vec, ord=(1 if tester.this_data.L1 else 2)) for vec in tester.vec_c]))
processes = [Process(target=test, args=(tester, index, rank_list, prec_list)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

rank_list = [x for x in rank_list]
rank_list.sort(key=lambda x: x[3])
with open(result_file, 'w') as fp:
    #head
    test_set = set([(h,r,t) for h,r,t in tester.test_triples])
    acc = len(set(prec_list) & test_set) * 1. / len(prec_list)
    prc = 0.
    prec = 0.
    num = 0.
    num_hit = 0
    total = total.value
    for h, r, t, _ in rank_list:
        hit = 0.
        if (h,r,t) in test_set:
            hit = 1.
            num_hit += 1
        prec = ((prec * num) + hit) / (num + 1)
        num += 1
        prc += prec
        if num_hit == total:
            break
    prc /= num
    print (acc, prc)
    fp.write("Accuracy=" + str(acc) + '  AUPRC=' + str(prc))