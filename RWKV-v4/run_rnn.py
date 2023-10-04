########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import math, os
import time
import types
import copy
import torch
from lightning_lite.utilities import data
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datetime import datetime

now = datetime.now() # current date and time

date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

from src.utils import GREBE_TOKENIZER, GrebeDataset

TEMPERATURE = 1.0
top_p = 0.7
top_p_newline = 0.9 # only used in TOKEN_MODE = char

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

########################################################################################################
# Step 1: set model
# 
# Set TOKEN_MODE to 'char' or 'bpe' if the model is trained by 'train.py' from scratch.
#
# Set TOKEN_MODE to 'pile' if you want to test pre-trained pile models.
########################################################################################################

TOKEN_MODE = 'pile'  # char / bpe / pile

n_layer = 6
n_embd = 512
ctx_len = 1024

if TOKEN_MODE == 'char':
    MODEL_NAME = 'trained-500'  # your trained model
    WORD_NAME = 'vocab'  # the .json vocab (generated by train.py)
    # set UNKNOWN_CHAR to the rarest token in your vocab.json, and all unknown tokens in your prompt will be denoted by it
    UNKNOWN_CHAR = ' '  # here we just set it to ' ' for simplicity

elif TOKEN_MODE == 'bpe':
    MODEL_NAME = 'trained-500'  # your trained model
    WORD_NAME = ['model-vocab.json', 'model-merges.txt']  # [vocab, merge] for your BPE model
    UNKNOWN_CHAR = None

elif TOKEN_MODE == 'pile':
    WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
    UNKNOWN_CHAR = None

    # ---> you can set MODEL_NAME to your fine-tuned model <---

    MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'
    #MODEL_NAME = 'RWKV-4-Pile-1B5-20220903-8040'
    # MODEL_NAME = 'trained-11'
    #n_layer = 24
    n_layer = 12
    #n_embd = 2048
    n_embd = 768
    ctx_len = 1024

    # MODEL_NAME = 'RWKV-4-Pile-430M-20220808-8066'
    # n_layer = 24
    # n_embd = 1024
    # ctx_len = 1024

    # MODEL_NAME = 'RWKV-4-Pile-1B5-20220903-8040'
    # n_layer = 24
    # n_embd = 2048
    # ctx_len = 1024    

os.environ['RWKV_FLOAT_MODE'] = 'fp32'  # 'bf16' / 'fp16' / 'fp32' (note: only using fp32 at this moment)
os.environ['RWKV_RUN_DEVICE'] = 'cpu'  # 'cpu' (already very fast) or 'cuda'
model_type = 'RWKV'  # 'RWKV' or 'RWKV-ffnPre'

ACTUAL_DEVICE = 'cuda'

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################

# context = 'A'
# context = "\nIn the"
# context = '\nSugar:'

NUM_TRIALS = 999
LENGTH_PER_TRIAL = 333

TEMPERATURE = 1.0
top_p = 0.7
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################

print(f'Loading {MODEL_NAME}...')

print("Here0")

from src.model_run import RWKV_RNN
from src.model_run import GREBE_RNN

#print("Here1")

model = GREBE_RNN(MODEL_NAME, ACTUAL_DEVICE, model_type, n_layer, n_embd, ctx_len, True)

#print("Here2")

tokenizer = GREBE_TOKENIZER(WORD_NAME)

import numpy as np

# load ascii text and covert to lowercase
filename = "../simplebooks/simplebooks-92/train.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()

tokenized = tokenizer.tokenizer.encode(raw_text)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, len(tokenized) - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append(seq_in)
    dataY.append(seq_out)

dataX = np.array(dataX)
dataY = np.array(dataY)

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

n_epochs = 40
batch_size = 1

loss_fn = torch.nn.CrossEntropyLoss()

#for token in tokenized:
#    print(tokenizer.tokenizer.decode(token))
#    batch_size += 1
#    if batch_size == 8:
#        break

dataset = GrebeDataset(tokenized, seq_length, len(tokenized), tokenizer.tokenizer)

train_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    torch.autograd.set_detect_anomaly(True)

    list_values = []
    moving_average = 0
    current_index = 0
    moving_amount = 100

    #for param in model.parameters():
    #    print("TEST1 ", param)
    #    print("TEST2 ", torch.sum(param))

    #print("XXX ", model.blocks0attreceptanceweight0)

    bef = 0

    for X_batch, y_batch in train_loader:

        y_pred = model(X_batch)
        print(torch.sum(y_pred))
        print(y_batch[0].float())

        char = tokenizer.sample_logits_bef(y_pred, bef, temperature=TEMPERATURE,
                                       top_p_usual=top_p, top_p_newline=top_p_newline)
        char = char.item()
        bef = char

        print("ANS ", tokenizer.tokenizer.decode(int(X_batch[0])), " + ", tokenizer.tokenizer.decode(int(char)), " + ", tokenizer.tokenizer.decode(torch.argmax(y_batch[0])))

        if ACTUAL_DEVICE == 'cuda':
            y_batch = y_batch.cuda()

        loss = loss_fn(y_pred.float(), y_batch[0].float())

        list_values.append(loss.item())

        if len(list_values) > moving_amount:
            list_values = list_values[1:]

        print("Current Step " + str(current_index) + " out of " + str(len(train_loader)))
        print("Current Percentage " + str(round((current_index / len(train_loader)) * 100, 2)) + "%")
        print("Loss: " + str(np.average(list_values)))

        print("EXAMPLE1: " + str(torch.sum(model.example1)))
        #print("EXAMPLE2: " + str(torch.sum(model.example2)))
        #print("EXAMPLE3: " + str(torch.sum(model.example3)))
        #print("EXAMPLE4: " + str(torch.sum(model.example4)))

        #for param in model.parameters():
        #    if param.grad:
        #        print("X ", param.grad.data.sum())


        #print("EXAMPLE2: " + str(torch.sum(model.example2)))

        current_index += 1
