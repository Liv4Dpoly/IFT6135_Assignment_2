# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:59:55 2019

@author: root2
"""
import os
import numpy as np
import matplotlib.pyplot as plt


problem_prefix = '4-3'
model_prefix = 'TRANSFORMER'
#title = "SGD_LR_SCHEDULE"
#title = "ADAM"
title = "TRANSFORMER"
#title = model_prefix

#problem_loc = os.path.join("C:\\","Users","root2","Desktop","Files","IFT 6135","Assignment 2","IFT6135",problem_prefix)
problem_loc = os.path.join ("loss")
full_path = os.path.join(problem_loc)
directories = os.listdir(full_path)

#pattern = model_prefix + "*"  
#for entry in directories:  
#    if fnmatch.fnmatch(entry, pattern):
##        print (entry)
##        print('\n'*2)
#        current_dir = os.path.join(full_path,entry)    
#        print (current_dir)
        
dir_list = []        
with os.scandir(full_path) as it:
    for entry in it:
        if entry.name.find(model_prefix) !=-1:
            dir_list.append(entry.name)
hyper_list = []
for combination in dir_list:
    try:
        x = np.load(os.path.join(full_path,combination,"learning_curves.npy"))[()]
    except:
        print(os.path.join(combination))
        continue
    train_ppls = x['train_ppls']
    val_ppls = x['val_ppls']
    train_losses = x['train_losses']
    val_losses = x['val_losses']
    average_loss = x['average_loss']
#    if np.any(np.isnan(val_ppls)) or np.any(np.isinf(val_ppls)) or np.sum(val_ppls)>10000:
#        print(combination)
#        continue
    epoch_num = len(train_ppls)
    epoch_num = 40
#_____________________________________________________________________________#
    #plt.plot(range(1,1+epoch_num), val_ppls[0:epoch_num ])
    plt.plot(range(1,1+epoch_num), average_loss[0:epoch_num ])
#    plt.legend(bbox_to_anchor=(0., 1.10, 1., .250), loc=3,
#       ncol=2, mode="expand", borderaxespad=0.)
#    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
#    plt.title(combination)
#    print(combination)
    hyper_list.append(combination)
plt.axis([1, 40, 0, 800])
plt.xlabel("Epoch")
#plt.ylabel("Validation PPL")
plt.ylabel("Average loss")
plt.title(title)
plt.savefig('val_ppls_over_Epochs'+model_prefix+'For_4-3.png')

plt.show()



#wall_time=[]
#num_lines = 40
#with open("log.txt", 'r') as f:
#    for line in f:
#        splitted = line.split()
#        wall_time.append(float(splitted[-1]))
#        num_lines += 1
#
#wall_clk_time = np.cumsum(wall_time)



for combination in dir_list:
    try:
        x = np.load(os.path.join(full_path,combination,"learning_curves.npy"))[()]
    except:
        print(os.path.join(combination))
        continue
    train_ppls = x['train_ppls']
    val_ppls = x['val_ppls']
    train_losses = x['train_losses']
    val_losses = x['val_losses']
    
    wall_time=[]
    num_lines = 40
    with open(os.path.join(full_path,combination,"log.txt"), 'r') as f:
        for line in f:
            splitted = line.split()
            wall_time.append(float(splitted[-1]))
            num_lines += 1
    
    wall_clk_time = np.cumsum(wall_time)
    
    
#    if np.any(np.isnan(val_ppls)) or np.any(np.isinf(val_ppls)) or np.sum(val_ppls)>10000:
#        print(combination)
#        continue
    epoch_num = len(train_ppls)
    epoch_num = 40
#_____________________________________________________________________________#
    plt.plot(wall_clk_time[0:min(len(wall_clk_time),epoch_num )], val_ppls[0:min(len(wall_clk_time),epoch_num ) ])
#    plt.legend(bbox_to_anchor=(0., 1.10, 1., .250), loc=3,
#       ncol=2, mode="expand", borderaxespad=0.)
#    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
#    plt.title(combination)
#    print(combination)
    hyper_list.append(combination)
plt.axis([0, 1600, 0, 800])
plt.xlabel("Time")
plt.ylabel("Validation PPL")
plt.title(title)
plt.savefig('val_ppls_over_Time'+model_prefix+'For_4-3.png')

plt.show()
















