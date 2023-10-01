import types
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score

os.environ['RWKV_RUN_DEVICE'] = 'cpu'

from src.model_run import RWKV_RNN
from src.utils import TOKENIZER


MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'
WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
DATA_FILE = '../winogrande_1.1/train_s.jsonl'
N_LAYER = 12
N_EMBD = 768
CTX_LEN = 1024


def load_winogrande_data(file_path):
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return data


def evaluate_example(model, tokenizer, example):
    sentence = example['sentence'].replace('_', '{}')
    options = [example['option1'], example['option2']]
    logits = []

    for option in options:
        context = "Who is referred to by the blank space? " + sentence.format(option) + " Who is referred to by the blank space?"
        ctx = tokenizer.tokenizer.encode(context)
        model.clear()

        init_state = types.SimpleNamespace()
        src_len = len(ctx)
        src_ctx = ctx.copy()

        for i in range(src_len):
            x = ctx[:i+1]
            if i == src_len - 1:
                init_state.out = model.run(x)
            else:
                model.run(x)

        model.save(init_state)
        model.load(init_state)

        logits.append(init_state.out[-1])

    return logits


def main():
    model = RWKV_RNN(MODEL_NAME, 'cpu', 'RWKV', N_LAYER, N_EMBD, CTX_LEN)
    tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=None)

    data = load_winogrande_data(DATA_FILE)

    predictions = []
    labels = []

    for example in data:
        logits = evaluate_example(model, tokenizer, example)
        prediction = np.argmax(logits) + 1
        ground_truth = example['answer']

        print(f"Question ID: {example['qID']}")
        print(f"Sentence: {example['sentence']}")
        print(f"Option 1: {example['option1']}, Option 2: {example['option2']}")
        print(f"Logits:  {logits}")
        print(f"Prediction: {prediction}, Ground Truth: {ground_truth} \n")

        predictions.append(str(prediction))
        labels.append(ground_truth)

    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
