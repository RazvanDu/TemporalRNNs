import os
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np

os.environ['RWKV_RUN_DEVICE'] = 'cuda'

from src.utils import TOKENIZER

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

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
N_LAYER = 12
N_EMBD = 768
N_PERSP = 4
DATA_FILE = '../ARC-Easy/ARC-Easy-Test.jsonl'
MODEL_PATH = './saves/ARC_Easy_10-08-2023-13-34-11'
CTX_LEN = 4096
BATCH_SIZE = 1
OUTPUT_FILE = './evaluation_logs/ARC_Easy_evaluation_results_' + current_time + '.txt'

if ours:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN, N_PERSP)
else:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN)

#model.load_state_dict(torch.load(MODEL_PATH))#
#model.eval()

tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=None)


def load_arc_easy_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


# Dataset
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
        label = ord(item['answerKey']) - ord('A')

        tokenized_choices = [self.tokenizer.tokenizer.encode(question + " " + choice) for choice in choices]

        return tokenized_choices, label, item['id'], question, choices


arc_dataset = ArcEasyDataset(load_arc_easy_data(DATA_FILE), tokenizer)
test_loader = DataLoader(arc_dataset, shuffle=False, batch_size=BATCH_SIZE)

softmax = torch.nn.Softmax(dim=0)
correct_count = 0
total_count = 0

with open(OUTPUT_FILE, 'w') as outfile:
    for i, (tokenized_choices, label, qID, question, choices) in enumerate(test_loader):
        logits_list = []

        for tokenized in tokenized_choices:
            sum_score = 0

            logits = []
            model.xx = {}
            model.aa = {}
            model.bb = {}
            model.pp = {}

            for j in range(len(tokenized) - 1):
                logits = model.run(tokenized[j])
                sum_score += logits[tokenized[j + 1]]

            logits_list.append(sum_score)

        #logits = torch.stack(logits_list, dim=0)
        #pred_label = torch.argmax(logits).item()

        pred_label = np.argmax(logits_list)

        if pred_label == label:
            correct_count += 1

        total_count += 1

        accuracy = (correct_count / total_count) * 100
        print(f"Evaluation complete. Accuracy: {accuracy}%\n")

        output_text = (
            f"Question number: {total_count}\{len(arc_dataset)}\n"
            f"Question ID: {qID}\n"
            f"Question: {question}\n"
            f"Choices: {choices}\n"
            f"Prediction: {chr(65 + pred_label)}, Ground Truth: {chr(65 + label)}\n\n"
            f"Current accuracy {accuracy}"
        )

        print(output_text)
        outfile.write(output_text)

    accuracy = (correct_count / total_count) * 100
    print(f"Evaluation complete. Accuracy: {accuracy}%\n")
    outfile.write(f"Evaluation complete. Accuracy: {accuracy}%\n")

print(f"Evaluation complete. Accuracy: {accuracy}%")
