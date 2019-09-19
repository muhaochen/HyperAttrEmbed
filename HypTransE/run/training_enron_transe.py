
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
import transe
import trainer_transe
from trainer_transe import Trainer

this_dim = 50
# In[4]:
model_path = './enron-transe.ckpt'
data_path = './enron-data.bin'
more_filt = ["../../../enron/gender_identified_enron_corpus/graph/partitions/structure_employee_wemail_random_test0.2.csv"]
filename = ["../../../enron/gender_identified_enron_corpus/graph/partitions/structure_employee_wemail_random_train0.8.csv", "../../../enron/gender_identified_enron_corpus/graph/structure_wemail.csv", "../../../enron/gender_identified_enron_corpus/graph/structure_wemail.csv"]


if len(sys.argv) > 1:
    this_dim = int(sys.argv[1])
    model_path = sys.argv[2]
    data_path = sys.argv[3]
    more_filt = [sys.argv[4]]
    filename = sys.argv[5:]

this_data = data.Data()
this_data.load_data_enron_csv(filenames=filename)
for f in more_filt:
    this_data.record_more_enron_csv(f)


# In[ ]:

m_train = Trainer()
m_train.build(this_data, dim=this_dim, batch_size=64, num_neg=10, save_path = model_path, data_save_path = data_path, L1=False)


# In[ ]:

m_train.train(epochs=1000, save_every_epoch=100, lr=0.001, a1=0., m1=1., burn_in=1)


# In[ ]:



