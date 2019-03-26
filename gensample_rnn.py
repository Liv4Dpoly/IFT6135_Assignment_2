import torch

import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy
from models import RNN, GRU 
from models import make_model as TRANSFORMER

#from ptb_lm import Batch, ptb_raw_data,_file_to_word_ids,_build_vocab,_read_words

generated_seq_len = 35
#generated_seq_len = 70
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#args_seed=1111
#torch.manual_seed(args_seed)

###############################################################################
#
# 
#LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    
    This prevents Pytorch from trying to backpropagate into previous input 
    sequences when we use the final hidden states from one mini-batch as the 
    initial hidden states for the next mini-batch.
    
    Using the final hidden states in this way makes sense when the elements of 
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)
###############################################################################
#
# 
#ARGUMENT PARAMETERS
#
###############################################################################
#Transformer
args_dp_keep_prob = 0.35
args_num_layers = 2
args_hidden_size = 512
model_batch_size = 20
model_seq_len =35
#RNN
args_emb_size_rnn = 200
args_hidden_size_rnn = 1500
args_seq_len_rnn = model_seq_len
args_batch_size_rnn = model_batch_size
args_num_layers_rnn = 2
args_dp_keep_prob_rnn = 0.35

###############################################################################
#
# 
# LOAD DATA
#
###############################################################################

args_data= './data'
print('Loading data from '+ args_data)
raw_data = ptb_raw_data(data_path=args_data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))

raw_data = np.array(test_data, dtype=np.int32)

data_len = len(raw_data)
batch_len = data_len // model_batch_size
data = np.zeros([model_batch_size, batch_len], dtype=np.int32)
for i in range(model_batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]



###############################################################################
#
# 
# RESTORE AND LOAD MODEL
#
###############################################################################

model = RNN(emb_size=args_emb_size_rnn, hidden_size=args_hidden_size_rnn, 
                seq_len=args_seq_len_rnn, batch_size=args_batch_size_rnn,
                vocab_size=vocab_size, num_layers=args_num_layers_rnn, 
                dp_keep_prob=args_dp_keep_prob_rnn) 


'''
model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size, 
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers, 
                dp_keep_prob=args.dp_keep_prob)

'''
'''
model = TRANSFORMER(vocab_size=vocab_size, n_units=args_hidden_size, 
                            n_blocks=args_num_layers, dropout=1.-args_dp_keep_prob)
'''                            
save_state_dict_path = './RNN_SGD_LR_SCHEDULE_model=RNN_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0/best_params.pt' 
pretrained_model = torch.load(save_state_dict_path,map_location=device)
model.load_state_dict(pretrained_model,strict=False)
model.to(device)
model.eval()
# if args.model != 'TRANSFORMER':
hidden = model.init_hidden()
hidden = hidden.to(device)
###############################################################################
#
# 
# GENERATE SAMPLES
#
###############################################################################
import os
if os.path.exists("outfile_rnn.txt"):
    os.remove("outfile_rnn.txt")
for s in range (20):
    epoch_size = (batch_len - 1) // model_seq_len
    i = torch.randint(epoch_size, (1, 1), dtype=torch.long)
    i.to(device)
    x = np.zeros((model_batch_size,model_seq_len))
    x[:,0] = data[:,i]

    x = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)

    with open('outfile_rnn.txt', 'a+') as outf:
        outf.write(('Sample ')+str(s+1)+(' : ')+('\n'))
        with torch.no_grad(): 
            for i in range (generated_seq_len):

                hidden = repackage_hidden(hidden)
                outputs, hidden = model(x, hidden)
           
                topv, topi = torch.topk(outputs[0],1,0)
                topimax = topi[0][0]
            
                word = id_2_word[topimax.item()]
                #print(word)
                tmp = x
                x[:,0:-1] = tmp[:,1:]
                x[0,-1] = topimax.item()  
            
                outf.write(word + ('\n' if i % 11 == 10 else ' '))

                if i % 20 == 0:
                    print('| Generated {}/{} words'.format(i, generated_seq_len))

        outf.write('\n')
