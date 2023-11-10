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
# MODEL_NAME = 'RWKV-4-Pile-1B5-20220903-8040'
if ours:
    MODEL_NAME = 'trained-35'
else:
    MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'

# Dataset location
OUTPUT_FILE = './evaluation_logs/PIQA_evaluation_results_' + current_time + '.txt'
VAL_DATA_FILEPATH = '../PIQA/dev.jsonl'
VAL_LABELS_FILEPATH = '../PIQA/dev-labels.lst'
TEST_DATA_FILEPATH = '../PIQA/test.jsonl'
PREDICTIONS_FILEPATH = 'PIQA_test_predictions.lst'

WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
# N_LAYER = 32
# N_EMBD = 2560
N_LAYER = 12
N_EMBD = 768
N_PERSP = 4
# MODEL_PATH = './saves/PIQA'
CTX_LEN = 4096
BATCH_SIZE = 1

if ours:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN, N_PERSP)
else:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN)

# model.load_state_dict(torch.load(MODEL_PATH))#
# model.eval()

tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=None)


def load_piqa_data(file_path, labels_file_path=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    if labels_file_path:
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f]
            for i, item in enumerate(data):
                item['label'] = labels[i]

    return data


class PIQADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        goal = item['goal']
        sol1 = item['sol1']
        sol2 = item['sol2']
        label = item.get('label', -1)  # Default label for test set is -1

        tokenized_goal = self.tokenizer.tokenizer.encode(goal)
        tokenized_sol1 = self.tokenizer.tokenizer.encode(sol1)
        tokenized_sol2 = self.tokenizer.tokenizer.encode(sol2)

        return tokenized_goal, [tokenized_sol1, tokenized_sol2], label, item['id'], goal, [sol1, sol2]


def evaluate_on_validation_set(data_loader, model):
    correct_predictions = 0
    total_predictions_count = 0

    with open(OUTPUT_FILE, 'w') as outfile:
        for i, (tokenized_goal, tokenized_solutions, true_label, qID, goal, solutions) in enumerate(data_loader):
            logits_list = []

            model.xx = {}
            model.aa = {}
            model.bb = {}
            model.pp = {}
            model.hk = {}

            for j in range(len(tokenized_goal)):
                logits_temp = model.run(tokenized_goal[j]).cpu()

            xx = copy.deepcopy(model.xx)
            aa = copy.deepcopy(model.aa)
            bb = copy.deepcopy(model.bb)
            pp = copy.deepcopy(model.pp)
            hk = copy.deepcopy(model.hk)

            for tokenized_solution in tokenized_solutions:
                model.xx = copy.deepcopy(xx)
                model.aa = copy.deepcopy(aa)
                model.bb = copy.deepcopy(bb)
                model.pp = copy.deepcopy(pp)
                model.hk = copy.deepcopy(hk)

                sum_score = 1
                logits = logits_temp.clone()
                for j in range(len(tokenized_solution)):
                    sum_score += logits.numpy().tolist()[tokenized_solution[j]]
                    logits = model.run(tokenized_solution[j]).cpu()

                logits_list.append(sum_score / len(tokenized_solution))

            pred_label = np.argmax(logits_list)

            if pred_label == true_label:
                correct_predictions += 1

            total_predictions_count += 1

            current_accuracy = (correct_predictions / total_predictions_count) * 100
            print(f"Evaluation in progress... Current accuracy: {current_accuracy}%\n")

            output_text = (
                f"Question number: {total_predictions_count}/{len(data_loader.dataset)}\n"
                f"Question ID: {qID}\n"
                f"Problem: {goal}\n"
                f"Solutions: {solutions}\n"
                f"Prediction: {'sol1' if pred_label == 0 else 'sol2'}, Ground Truth: {'sol1' if true_label == 0 else 'sol2'}\n"
                f"Current accuracy: {current_accuracy}%\n\n"
            )

            print(output_text)
            outfile.write(output_text)

        accuracy = (correct_predictions / total_predictions_count) * 100
        print(f"Evaluation complete. Accuracy: {accuracy}%\n")
        outfile.write(f"Evaluation complete. Accuracy: {accuracy}%\n")

    print(f"Evaluation complete. Accuracy: {accuracy}%")


def generate_predictions_for_piqa_test_set(data_loader, model):
    with open(PREDICTIONS_FILEPATH, 'w') as pred_file:
        for i, (tokenized_goal, tokenized_solutions, _, qID, goal, solutions) in enumerate(data_loader):
            logits_list = []

            model.xx = {}
            model.aa = {}
            model.bb = {}
            model.pp = {}
            model.hk = {}

            for j in range(len(tokenized_goal)):
                logits_temp = model.run(tokenized_goal[j]).cpu()

            xx = copy.deepcopy(model.xx)
            aa = copy.deepcopy(model.aa)
            bb = copy.deepcopy(model.bb)
            pp = copy.deepcopy(model.pp)
            hk = copy.deepcopy(model.hk)

            for tokenized_solution in tokenized_solutions:
                model.xx = copy.deepcopy(xx)
                model.aa = copy.deepcopy(aa)
                model.bb = copy.deepcopy(bb)
                model.pp = copy.deepcopy(pp)
                model.hk = copy.deepcopy(hk)

                sum_score = 1
                logits = logits_temp.clone()
                for j in range(len(tokenized_solution)):
                    sum_score += logits.numpy().tolist()[tokenized_solution[j]]
                    logits = model.run(tokenized_solution[j]).cpu()

                logits_list.append(sum_score / len(tokenized_solution))

            pred_label = np.argmax(logits_list)

            pred_file.write(f'{pred_label}\n')

    print('Test set predictions saved to', PREDICTIONS_FILEPATH)


val_dataset = PIQADataset(load_piqa_data(VAL_DATA_FILEPATH, VAL_LABELS_FILEPATH), tokenizer)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = PIQADataset(load_piqa_data(TEST_DATA_FILEPATH), tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

evaluate_on_validation_set(val_loader, model)

print('Evaluation done for the validation set!')

generate_predictions_for_piqa_test_set(test_loader, model)

print('Preidctions generated for the test set!')
