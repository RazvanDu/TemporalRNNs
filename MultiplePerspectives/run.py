import numpy as np
import math, os
import time
import types
import copy
import torch
from torch.nn import functional as F
from src.utils import TOKENIZER, Dataset
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

TOKEN_MODE = 'pile'

ours = True

if TOKEN_MODE == 'char':
    MODEL_NAME = 'trained-500'
    WORD_NAME = 'vocab'       
    UNKNOWN_CHAR = ' '      

elif TOKEN_MODE == 'bpe':
    MODEL_NAME = 'trained-500' 
    WORD_NAME = ['model-vocab.json', 'model-merges.txt'] 
    UNKNOWN_CHAR = None

elif TOKEN_MODE == 'pile':
    WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
    UNKNOWN_CHAR = None

    if ours:
        MODEL_NAME = 'trained-small-2'
    else:
        MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'

    N_PERSP = 4
    N_LAYER = 12
    N_EMBD = 768
    CTX_LEN = 1024

    # MODEL_NAME = 'RWKV-4-Pile-430M-20220808-8066'
    # n_layer = 24
    # n_embd = 1024
    # ctx_len = 1024

    # MODEL_NAME = 'RWKV-4-Pile-1B5-20220903-8040'
    # n_layer = 24
    # n_embd = 2048
    # ctx_len = 1024    

os.environ['RWKV_FLOAT_MODE'] = 'fp32'
os.environ['RWKV_RUN_DEVICE'] = 'cpu' 
model_type = 'RWKV' 

ctx_len = CTX_LEN

context = "\n"

NUM_TRIALS = 999
LENGTH_PER_TRIAL = 333

TEMPERATURE = 1.0
top_p = 0.7
top_p_newline = 0.9

DEBUG_DEBUG = False 


print(f'Loading {MODEL_NAME}...')
if ours:
    from src.model_run_ours import RWKV_RNN
    model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, N_LAYER, N_EMBD, CTX_LEN, N_PERSP)
else:
    from src.model_run import RWKV_RNN
    model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, N_LAYER, N_EMBD, CTX_LEN)

tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

if tokenizer.charMode:
    context = tokenizer.refine_context(context)
    ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
else:
    ctx = tokenizer.tokenizer.encode(context)
src_len = len(ctx)
src_ctx = ctx.copy()

print('\nYour prompt has ' + str(src_len) + ' tokens.')
print('\n--> Currently the first run takes a while if your prompt is long, as we are using RNN to process the prompt. Use GPT to build the hidden state for better speed. <--\n')

for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    t_begin = time.time_ns()
    print(('-' * 30) + context, end='')
    ctx = src_ctx.copy()
    model.clear()
    if TRIAL == 0:
        init_state = types.SimpleNamespace()
        for i in range(src_len):
            x = ctx[:i+1]
            print(tokenizer.tokenizer.decode(x[-1]), end="")
            if i == src_len - 1:
                init_state.out = model.run(x)
            else:
                model.run(x)
        model.save(init_state)
    else:
        model.load(init_state)

    break
    for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
        x = ctx[:i+1]
        x = x[-ctx_len:]

        if i == src_len:
            out = copy.deepcopy(init_state.out)
        else:
            out = model.run(x)
        if DEBUG_DEBUG:
            print('model', np.array(x), '==>', np.array(
                out), np.max(out), np.min(out))

        if TOKEN_MODE == 'pile':
            out[0] = -999999999 

        char = tokenizer.sample_logits(out, x, ctx_len, temperature=TEMPERATURE,
                                       top_p_usual=top_p, top_p_newline=top_p_newline)
        char = char.item()
        if tokenizer.charMode:
            print(tokenizer.itos[int(char)], end='', flush=True)
        else:
            print(tokenizer.tokenizer.decode(int(char)), end='', flush=True)
        ctx += [char]

    t_end = time.time_ns()
    print("\n----------", round((t_end - t_begin) / (10 ** 9), 2), end='s ')
