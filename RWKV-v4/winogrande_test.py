import os
import json
import torch
from torch import optim, nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F
import gc
os.environ['RWKV_RUN_DEVICE'] = 'cpu'
from datetime import datetime

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

SEED = 42

now = datetime.now() # current date and time

date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
# Your existing imports

from src.utils import TOKENIZER

ours = True

if ours:
    from src.model_run_ours import RWKV_RNN
else:
    from src.model_run import RWKV_RNN

# Define constants
#MODEL_NAME = 'RWKV-4-Pile-1B5-20220903-8040'
if ours:
    MODEL_NAME = 'trained-10'
else:
    MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'
WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
DATA_FILE = '../winogrande_1.1/dev.jsonl'
N_LAYER = 12
N_EMBD = 768
N_PERSP = 4
#N_LAYER = 32
#N_EMBD = 2560
CTX_LEN = 4096
#CTX_LEN = 1024
SEQ_LEN = 100  # You may adjust this
BATCH_SIZE = 1  # You may adjust this

aug = naw.RandomWordAug(action="swap")

np.random.seed(SEED)
# Initialize model and tokenizer
if ours:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN, N_PERSP)
else:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN)
#model = GREBE_RNN(MODEL_NAME, 'cpu', 'RWKV', N_LAYER, N_EMBD, CTX_LEN)
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=None)

def print_tensors_in_memory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except Exception as e:
            pass

def load_winogrande_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


class WinograndeDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        sentence = item['sentence']
        option1, option2 = item['option1'], item['option2']

        tokenized1 = tokenizer.tokenizer.encode(sentence.replace("_", option1))
        tokenized2 = tokenizer.tokenizer.encode(sentence.replace("_", option2))

        label = int(item['answer']) - 1  # Converting 1-indexed to 0-indexed

        print("QUESTION ", sentence)
        print("OPT1 ", option1)
        print("OPT2 ", option2)

        return tokenized1, tokenized2, label

dataset = WinograndeDataset(load_winogrande_data(DATA_FILE), tokenizer)
test_loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

# Training loop
n_epochs = 1

softmax = nn.Softmax(dim=0)

for epoch in range(n_epochs):

    #torch.autograd.set_detect_anomaly(True)

    loss_list = []
    acc_list = []
    acc_list_total = []

    for i, (tokenized1, tokenized2, label) in enumerate(test_loader):

        sum1 = 0
        sum2 = 0

        logits = []
        model.xx = {}
        model.aa = {}
        model.bb = {}
        model.pp = {}

        for j in range(len(tokenized1)-1):
            #print("q ", len(model.run(tokenized1[j+1])))
            logits = model.run(tokenized1[j])
            sum1 += logits[tokenized1[j+1]]
            #print("X ", torch.log(logits[tokenized1[j+1]]), " - ", logits[tokenized1[j+1]])

        logits = []
        model.xx = {}
        model.aa = {}
        model.bb = {}
        model.pp = {}

        for j in range(len(tokenized2)-1):
            logits = model.run(tokenized2[j])
            sum2 += logits[tokenized2[j+1]]

        if sum1 > sum2:
            opt = 0
        else:
            opt = 1

        print("O ", opt)
        print("L ", label)

        if opt == label:
            acc_list.append(1)
            print("CORRECT")
        else:
            acc_list.append(0)
            print("WRONG")

        acc_list_total.append(acc_list[-1])

        print("Step: ", i, "/", len(test_loader))

        print(f"Epoch {epoch + 1}, Iteration {i}, Loss: {np.average(loss_list)}")
        print("Running accuracy ", np.sum(acc_list)/len(acc_list))
        print("Total accuracy ", np.sum(acc_list_total)/len(acc_list_total))

print("Testing complete")
