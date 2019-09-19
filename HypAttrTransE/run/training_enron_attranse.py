
# coding: utf-8

# In[1]:



# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[3]:

import sys
if '../src' not in sys.path:
    sys.path.append('../src')

if '../../utils' not in sys.path:
    sys.path.append('../../utils')
    
import numpy as np
import os
import tensorflow as tf

import data  # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import attranse
import trainer_attranse
from trainer_attranse import Trainer

this_dim = 50
this_b = 1.
this_c = 1.
# In[4]:
model_path = './enron-attranse.ckpt'
data_path = './enron-data.bin'
more_filt = ["../../../enron/gender_identified_enron_corpus/graph/partitions/structure_employee_wemail_random_test0.2.csv"]
feature_path = "../../../enron/gender_identified_enron_corpus/graph/attributes/emp_aggr_email_attr_norm.csv"
filename = ["../../../enron/gender_identified_enron_corpus/graph/partitions/structure_employee_wemail_random_train0.8.csv"]


if len(sys.argv) > 1:
    this_dim = int(sys.argv[1])
    this_b = float(sys.argv[2])
    this_c = float(sys.argv[3])
    model_path = sys.argv[4]
    data_path = sys.argv[5]
    more_filt = [sys.argv[6]]
    feature_path = sys.argv[7]
    filename = sys.argv[8:]

this_data = data.Data()
this_data.load_data_enron_csv(filenames=filename)
for f in more_filt:
    this_data.record_more_enron_csv(f)

this_data.load_p2p_attr_enron_csv(feature_path, 11)
this_data.train_attr_svm()
this_data.calculate_pp2c(this_data.triples_record)
# In[ ]:

m_train = Trainer()
m_train.build(this_data, dim=this_dim, batch_size=128, save_path = model_path, data_save_path = data_path, L1=False)


# In[ ]:

m_train.train(epochs=100, save_every_epoch=100, lr=0.001, a1=0., m1=1., b = this_b, c = this_c)


# In[ ]:



