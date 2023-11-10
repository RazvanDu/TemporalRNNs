import os
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
import copy

os.environ['RWKV_RUN_DEVICE'] = 'cuda'

from src.utils import TOKENIZER

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

ours = False

if ours:
    from src.model_run_ours import RWKV_RNN
else:
    from src.model_run import RWKV_RNN

# Define constants
if ours:
    MODEL_NAME = 'trained-40'
else:
    MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'

WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
# N_LAYER = 32
# N_EMBD = 2560
N_LAYER = 12
N_EMBD = 768
N_PERSP = 4
DATA_FILE = '../SciQ/test.json'
OUTPUT_FILE = './evaluation_logs/SciQ_evaluation_results_' + current_time + '.txt'
CTX_LEN = 4096
BATCH_SIZE = 1

if ours:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN, N_PERSP)
else:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN)

tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=None)


def load_sciq_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data_json = json.load(file)

    data = []
    for item in data_json:
        question = item['question']
        correct_answer = item['correct_answer']
        distractors = [item['distractor1'], item['distractor2'], item['distractor3']]
        choices = distractors + [correct_answer]
        label = choices.index(correct_answer)

        data.append({
            'question': question,
            'choices': choices,
            'label': label,
            'support': item.get('support', '')
        })
    return data


class SciQDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        choices = item['choices']
        label = item['label']
        support = item['support']

        tokenized_question = self.tokenizer.tokenizer.encode(question)
        tokenized_choices = [self.tokenizer.tokenizer.encode(choice) for choice in choices]

        return tokenized_question, tokenized_choices, label, question, choices, support


sci_dataset = SciQDataset(load_sciq_data(DATA_FILE), tokenizer)
test_loader = DataLoader(sci_dataset, shuffle=False, batch_size=BATCH_SIZE)

correct_count = 0
total_count = 0

with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    for i, (tokenized_question, tokenized_choices, label, question, choices, support) in enumerate(test_loader):
        logits_list = []

        model.xx = {}
        model.aa = {}
        model.bb = {}
        model.pp = {}
        model.hk = {}

        for j in range(len(tokenized_question)):
            logits_temp = model.run(tokenized_question[j]).cpu()
        xx = copy.deepcopy(model.xx)
        aa = copy.deepcopy(model.aa)
        bb = copy.deepcopy(model.bb)
        pp = copy.deepcopy(model.pp)
        hk = copy.deepcopy(model.hk)

        for tokenized in tokenized_choices:
            model.xx = copy.deepcopy(xx)
            model.aa = copy.deepcopy(aa)
            model.bb = copy.deepcopy(bb)
            model.pp = copy.deepcopy(pp)
            model.hk = copy.deepcopy(hk)

            sum_score = 1
            logits = logits_temp.clone()
            for j in range(len(tokenized)):
                sum_score += logits.numpy().tolist()[tokenized[j]]
                logits = model.run(tokenized[j]).cpu()

            logits_list.append(sum_score / len(tokenized))

        pred_label = np.argmax(logits_list)

        if pred_label == label:
            correct_count += 1
        total_count += 1

        accuracy = (correct_count / total_count) * 100
        print(f"Evaluation complete. Accuracy: {accuracy}%\n")

        output_text = (
            f"Question number: {total_count}/{len(sci_dataset)}\n"
            f"Question ID: {i+1}\n"
            f"Question: {question}\n"
            f"Choices: {choices}\n"
            f"Support: {support}\n"
            f"Prediction: {chr(65 + pred_label)}, Ground Truth: {chr(65 + label)}\n"
            f"Current accuracy: {accuracy}%\n\n"
        )

        print(output_text)
        outfile.write(output_text)

    accuracy = (correct_count / total_count) * 100
    print(f"Evaluation complete. Accuracy: {accuracy}%")
    outfile.write(f"Evaluation complete. Accuracy: {accuracy}%\n")

print(f"Evaluation complete. Accuracy: {accuracy}%")
