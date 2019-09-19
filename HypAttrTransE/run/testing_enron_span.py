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

rels = set([x[1] for x in tester.test_triples])

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()

rank_list = []
for x in range(len(tester.span)):
    rank_list.append([])

index = 9 #index
total = 0 #index
select_per_span = np.zeros(len(tester.span), dtype=np.int32)
for h,r,t in tester.test_triples:
    select_per_span[tester.e2span[h]] += 1 # tester.e2span[h] == tester.e2span[t]

t0 = time.time()

def test(tester, index, rank_list):
    while index < len(tester.span):
        id = index
        index += 1
        if id % 5 == 0:
            print("Tested %d spans in %d seconds." % (id+1, time.time()-t0))
        span = tester.span[id]
        if span is None:
            continue
        print(span)
        for h in span:
            for t in span:
                if h == t:
                    continue
                for r in rels:
                    dis = tester.dissimilarity(h, r, t)
                    rank_list[id].append((h,r,t,dis))
        #Change

print('Average norm=%d' % np.mean([LA.norm(vec, ord=(1 if tester.this_data.L1 else 2)) for vec in tester.vec_c]))
test(tester, index, rank_list)



for i in range(len(rank_list)):
    rank_list[i].sort(key=lambda x: x[3])
rst = []
for i in range(len(rank_list)):
    if len(rank_list[i]) > 0:
        rst += [(h,r,t) for h,r,t,_ in rank_list[i][:select_per_span[i]]]
with open(result_file, 'w') as fp:
    #head
    test_set = set([(h,r,t) for h,r,t in tester.test_triples])
    acc = len(set(rst) & test_set) * 1. / len(rst)
    print (acc)
    fp.write("Accuracy=" + str(acc))