# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:09:18 2019

@author: root2
"""
import numpy as np
import matplotlib.pyplot as plt

path_loss = './loss/TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.001_batch_size=196_seq_len=45_hidden_size=256_num_layers=2_dp_keep_prob=.9_save_best_0/losses_4.npy'
x = np.load(path_loss)[()]
plt.plot(range(1,46), x['losses'])
plt.axis([1,45, 3.5, 7])
plt.title('TRANSFORMER')
plt.xlabel("Time step")
plt.ylabel("Average loss")
#plt.savefig('Average_loss_GRU.png')
plt.savefig('Average_loss_Transformer.png')
