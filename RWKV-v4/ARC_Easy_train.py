import gc
import json
import numpy as np
import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

os.environ['RWKV_RUN_DEVICE'] = 'cpu'
from datetime import datetime

now = datetime.now()  # current date and time

date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
# Your existing imports
from src.model_run import GREBE_RNN
from src.utils import TOKENIZER

# Define constants
# MODEL_NAME = 'RWKV-4-Pile-1B5-20220903-8040'
MODEL_NAME = 'RWKV-4-Pile-430M-20220808-8066'
WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
DATA_FILE = '../ARC-Easy/ARC-Easy-Train.jsonl'
N_LAYER = 24
N_EMBD = 1024
# N_LAYER = 32
# N_EMBD = 2560
CTX_LEN = 4096
# CTX_LEN = 1024
SEQ_LEN = 100  # You may adjust this
BATCH_SIZE = 1  # You may adjust this

# Initialize model and tokenizer
model = GREBE_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN, None)
# model = GREBE_RNN(MODEL_NAME, 'cpu', 'RWKV', N_LAYER, N_EMBD, CTX_LEN)
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=None)


def print_tensors_in_memory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except Exception as e:
            pass


def load_arc_easy_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


class ArcEasyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']['stem']
        choices = [choice['text'] for choice in item['question']['choices']]
        label = ord(item['answerKey']) - ord('A')  # Converting from 'A', 'B', 'C', 'D' to 0, 1, 2, 3

        tokenized_choices = [self.tokenizer.tokenizer.encode(question + " " + choice) for choice in choices]

        return tokenized_choices, label


arc_dataset = ArcEasyDataset(load_arc_easy_data(DATA_FILE), tokenizer)
train_loader = DataLoader(arc_dataset, shuffle=True, batch_size=BATCH_SIZE)

optimizer = optim.Adam(model.parameters(), lr=0.00004)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
n_epochs = 100

softmax = nn.Softmax(dim=0)

for epoch in range(n_epochs):

    model.train()
    # torch.autograd.set_detect_anomaly(True)

    loss_list = []
    acc_list = []
    acc_list_total = []

    for i, (tokenized_choices, label) in enumerate(train_loader):

        if i >= 20 and epoch != n_epochs - 1:
            break

        if i == 20:
            loss_list = []
            acc_list = []
            acc_list_total = []

        logits_list = []

        for tokenized in tokenized_choices:
            sum_score = 0
            model.xx = {}
            model.aa = {}
            model.bb = {}
            model.pp = {}

            for j in range(len(tokenized) - 1):
                logits = softmax(model(tokenized[j]))
                sum_score += torch.log(logits[tokenized[j + 1]])

            logits_list.append(sum_score)

        logits = torch.stack(logits_list, dim=0)

        print("LO ", logits)

        loss = loss_fn(logits.unsqueeze(0), label.cuda().unsqueeze(0))

        if i < 20:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_label = torch.argmax(logits).item()

        if pred_label == label:
            acc_list.append(1)
        else:
            acc_list.append(0)

        acc_list_total.append(acc_list[-1])

        print("EXAMPLE1: " + str(torch.sum(model.example1)))
        print("EXAMPLE2: " + str(torch.sum(model.example2)))
        print("EXAMPLE3: " + str(torch.sum(model.example3)))
        # print("EXAMPLE4: " + str(torch.sum(model.example4)))

        print("Step: ", i, "/", len(train_loader))

        loss_list.append(loss.item())

        if len(loss_list) > 100:
            loss_list.pop(0)
        if len(acc_list) > 100:
            acc_list.pop(0)

        print(f"Epoch {epoch + 1}, Iteration {i}, Loss: {np.average(loss_list)}")
        print("Running accuracy ", np.sum(acc_list) / len(acc_list))
        print("Total accuracy ", np.sum(acc_list_total) / len(acc_list_total))

        if i % 50 == 0:
            torch.save(model.state_dict(), 'saves/ARC_Easy_' + str(date_time))

        torch.cuda.empty_cache()

print("Training complete")
