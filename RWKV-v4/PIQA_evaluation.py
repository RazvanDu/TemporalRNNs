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

        return tokenized_goal, tokenized_sol1, tokenized_sol2, label, item['id'], goal, [sol1, sol2]


def evaluate_on_validation_set(data_loader, model, tokenizer):
    correct_predictions = 0
    total_predictions_count = 0

    with open(OUTPUT_FILE, 'w') as outfile:
        for tokenized_goal, tokenized_sol1, tokenized_sol2, true_label, qID, goal, solutions in data_loader:
            # TODO: Replace with the correct evaluation, current solution doesn't work
            sol1_score = model.run(tokenized_goal + tokenized_sol1).cpu()
            sol2_score = model.run(tokenized_goal + tokenized_sol2).cpu()

            predicted_label = 0 if sol1_score > sol2_score else 1
            correct = predicted_label == true_label

            correct_predictions += correct
            total_predictions_count += 1

            current_accuracy = correct_predictions / total_predictions_count

            output_text = (
                f"Question ID: {qID}\n"
                f"Question: {goal}\n"
                f"Solutions: {solutions}\n"
                f"Prediction: {'sol1' if predicted_label == 0 else 'sol2'}, "
                f"Actual: {'sol1' if true_label == 0 else 'sol2'}, "
                f"Correct: {correct}\n"
                f"Current accuracy: {current_accuracy}\n\n"
            )

            print(output_text)
            outfile.write(output_text)

        accuracy = correct_predictions / len(data_loader.dataset)
        print(f'Validation Evaluation complete. Accuracy: {accuracy:.2%}\n')
        outfile.write(f'Validation Evaluation complete. Accuracy: {accuracy:.2%}\n')


val_dataset = PIQADataset(load_piqa_data(VAL_DATA_FILEPATH, VAL_LABELS_FILEPATH), tokenizer)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = PIQADataset(load_piqa_data(TEST_DATA_FILEPATH), tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

evaluate_on_validation_set(val_loader, model, tokenizer)

# We don't have the ground truth for the test set, we save the predictions to a file and we need to submit to a contest page (I think)
with open(PREDICTIONS_FILEPATH, 'w') as pred_file:
    for tokenized_goal, tokenized_sol1, tokenized_sol2, _, qID, goal, solutions in test_loader:
        # TODO: Replace with the correct evaluation, current solution doesn't work
        sol1_score = model.run(tokenized_goal + tokenized_sol1).cpu()
        sol2_score = model.run(tokenized_goal + tokenized_sol2).cpu()

        predicted_label = 0 if sol1_score > sol2_score else 1

        pred_file.write(f'{predicted_label}\n')

print('Test set predictions saved to', PREDICTIONS_FILEPATH)
