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
OUTPUT_FILE = './evaluation_logs/HellaSwag_evaluation_results_' + current_time + '.txt'
VAL_DATA_FILEPATH = '../HellaSwag/valid.jsonl'
VAL_LABELS_FILEPATH = '../HellaSwag/valid-labels.lst'
TEST_DATA_FILEPATH = '../HellaSwag/test.jsonl'
PREDICTIONS_FILEPATH = 'HellaSwag_test_predictions.lst'

WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
# N_LAYER = 32
# N_EMBD = 2560
N_LAYER = 12
N_EMBD = 768
N_PERSP = 4
# MODEL_PATH = './saves/HellaSwag'
CTX_LEN = 4096
BATCH_SIZE = 1

if ours:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN, N_PERSP)
else:
    model = RWKV_RNN(MODEL_NAME, 'cuda', 'RWKV', N_LAYER, N_EMBD, CTX_LEN)

# model.load_state_dict(torch.load(MODEL_PATH))#
# model.eval()

tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=None)


def load_hellaswag_data(file_path, labels_file_path=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    if labels_file_path:
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f]
            for i, item in enumerate(data):
                item['label'] = labels[i]

    return data


class HellaSwagDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['ctx']
        ending_options = item['ending_options']
        label = item.get('label', -1)
        qID = str(item['ind'])

        tokenized_context = self.tokenizer.tokenizer.encode(context)
        tokenized_endings = [self.tokenizer.tokenizer.encode(ending) for ending in ending_options]

        return tokenized_context, tokenized_endings, label, qID, context, ending_options


def evaluate_on_validation_set(data_loader, model):
    correct_predictions = 0
    total_predictions_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for i, (tokenized_context, tokenized_endings, true_label, qID, context, endings) in enumerate(data_loader):
            logits_list = []

            model.xx = {}
            model.aa = {}
            model.bb = {}
            model.pp = {}
            model.hk = {}

            for token in tokenized_context:
                logits_temp = model.run(token).cpu()

            xx = copy.deepcopy(model.xx)
            aa = copy.deepcopy(model.aa)
            bb = copy.deepcopy(model.bb)
            pp = copy.deepcopy(model.pp)
            hk = copy.deepcopy(model.hk)

            for tokenized_ending in tokenized_endings:
                model.xx = copy.deepcopy(xx)
                model.aa = copy.deepcopy(aa)
                model.bb = copy.deepcopy(bb)
                model.pp = copy.deepcopy(pp)
                model.hk = copy.deepcopy(hk)

                sum_score = 1
                logits = logits_temp.clone()
                for token in tokenized_ending:
                    sum_score += logits.numpy().tolist()[token]
                    logits = model.run(token).cpu()

                logits_list.append(sum_score / len(tokenized_ending))

            pred_label = np.argmax(logits_list)

            if pred_label == true_label:
                correct_predictions += 1

            total_predictions_count += 1

            current_accuracy = (correct_predictions / total_predictions_count) * 100
            print(f"Evaluation in progress... Current accuracy: {current_accuracy}%\n")

            output_text = (
                f"Question number: {total_predictions_count}/{len(data_loader.dataset)}\n"
                f"Question ID: {str(qID)}\n"
                f"Context: {context}\n"
                f"Ending Options: {endings}\n"
                f"Prediction: {endings[pred_label]}, Ground Truth: {endings[true_label]}\n"
                f"Current accuracy: {current_accuracy}%\n\n"
            )

            print(output_text)
            outfile.write(output_text)

        accuracy = (correct_predictions / total_predictions_count) * 100
        print(f"Evaluation complete. Accuracy: {accuracy}%")
        outfile.write(f"Evaluation complete. Accuracy: {accuracy}%\n")


def generate_predictions_for_hellaswag_test_set(data_loader, model):
    with open(PREDICTIONS_FILEPATH, 'w', encoding='utf-8') as pred_file:
        for i, (tokenized_context, tokenized_endings, _, qID, context, endings) in enumerate(data_loader):
            logits_list = []

            model.xx = {}
            model.aa = {}
            model.bb = {}
            model.pp = {}
            model.hk = {}

            for token in tokenized_context:
                logits_temp = model.run(token).cpu()

            xx = copy.deepcopy(model.xx)
            aa = copy.deepcopy(model.aa)
            bb = copy.deepcopy(model.bb)
            pp = copy.deepcopy(model.pp)
            hk = copy.deepcopy(model.hk)

            for tokenized_ending in tokenized_endings:
                model.xx = copy.deepcopy(xx)
                model.aa = copy.deepcopy(aa)
                model.bb = copy.deepcopy(bb)
                model.pp = copy.deepcopy(pp)
                model.hk = copy.deepcopy(hk)

                sum_score = 1
                logits = logits_temp.clone()
                for token in tokenized_ending:
                    sum_score += logits.numpy().tolist()[token]
                    logits = model.run(token).cpu()

                logits_list.append(sum_score / len(tokenized_ending))

            pred_label = np.argmax(logits_list)

            pred_file.write(f'{pred_label}\n')

    print('Test set predictions saved to', PREDICTIONS_FILEPATH)


val_dataset = HellaSwagDataset(load_hellaswag_data(VAL_DATA_FILEPATH, VAL_LABELS_FILEPATH), tokenizer)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

test_dataset = HellaSwagDataset(load_hellaswag_data(TEST_DATA_FILEPATH), tokenizer)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

evaluate_on_validation_set(val_loader, model)

print('Evaluation done for the validation set!')

generate_predictions_for_hellaswag_test_set(test_loader, model)

print('Preidctions generated for the test set!')