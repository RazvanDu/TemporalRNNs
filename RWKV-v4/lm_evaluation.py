########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
#
import os, sys, types, json, math, time
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F
import torch.nn as nn

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
device = 'cuda'

from lm_evaluation_harness.lm_eval import tasks, evaluator
from lm_evaluation_harness.lm_eval.models.gpt2 import GPT2LM
from src import model_ours
from src import model as modell

ours = True

vocab_size = 50277
ctx_len = 1024
model_type = 'RWKV'
n_layer = 12
n_embd = 768
n_persp = 4

########################################################################################################

MODEL_NAME = "/fsx/BlinkDL/HF-MODEL/rwkv-5-world/RWKV-5-World-1.5B-v2-OnlyForTest_14%_trained-20231001-ctx4096"

print(f'Loading model - {MODEL_NAME}')
if ours:
    model = model_ours.GPT(model_ours.GPTConfig(vocab_size, ctx_len, model_type=model_type,
                                                n_layer=n_layer, n_embd=n_embd, n_persp=n_persp))
else:
    model = modell.GPT(modell.GPTConfig(vocab_size, ctx_len, model_type=model_type,
                                        n_layer=n_layer, n_embd=n_embd,
                                        n_persp=n_persp))
if ours:
    MODEL_NAME = 'wikipedia_trained/' + 'trained-24'
else:
    MODEL_NAME = 'weights/' + 'RWKV-4-Pile-169M-20220807-8023'

with torch.no_grad():
    print('loading', MODEL_NAME)
    m2 = torch.load(MODEL_NAME + '.pth', map_location='cpu')
    model.load_state_dict(m2)
    for param in model.state_dict():
        model.state_dict()[param].requires_grad = False
    del m2
model.to(device)
#pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

eval_tasks = []
# eval_tasks += ['arc_challenge','arc_easy','headqa','openbookqa',]
# eval_tasks += ['arc_challenge','arc_easy','headqa','openbookqa',]
eval_tasks += ['arc_easy', 'lambada_openai', 'piqa', 'sciq']#,'piqa','hellaswag','winogrande','triviaqa','sciq']
#eval_tasks += ['arc_easy']
# eval_tasks += ['lambada_openai']
# eval_tasks += ['hellaswag','winogrande']
# eval_tasks += ['lambada_openai','piqa','storycloze_2016','hellaswag','winogrande']
# eval_tasks += ['record','copa']
# eval_tasks += ['triviaqa']
# eval_tasks += ['coqa']

RWKV_PAD = tokenizer.encode('\n')  # we will use '\n' as PAD
# RWKV_PAD = [0] # you can try using [0] as pad
print('RWKV_PAD', RWKV_PAD)

########################################################################################################

logitBuf = {}
correctBuf = {}


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


class EvalHarnessAdapter(GPT2LM):
    def __init__(self, model):
        #self._max_length = 1024
        #self._device = 'cuda'
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.model = model
        #self.model = model

    # def greedy_until(self, requests): # designed for coqa
    #     res = []
    #     for i in range(len(requests)):
    #         if i % 50 == 0:
    #             print(i)
    #         otoken = []
    #         while True:
    #             src = self.tokenizer.encode(requests[i][0]) + otoken

    #             src = src[-4096:]
    #             outputs, _ = model.forward(src, None)

    #             otoken += [int(torch.argmax(outputs))]
    #             ss = self.tokenizer.decode(otoken)
    #             if '\n' in ss or len(ss) > 200:
    #                 if not ss.endswith('\n'):
    #                     ss = ss + '\n'
    #                 print(ss)
    #                 res += [(ss)]
    #                 break
    #     print(res)
    #     return res

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        global logitBuf, correctBuf

        res = []

        for COUNTER in range(len(requests)):
            n = COUNTER
            raw_src = requests[n][0][0] + requests[n][0][1]

            src = requests[n][1] + requests[n][2]

            raw_src = '\n' + raw_src
            src = RWKV_PAD + src

            sss = str(src)
            correct = True
            if sss in logitBuf:
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                q_len = len(requests[n][1])
                q_len += len(RWKV_PAD)
                logit = 0

                with torch.no_grad():
                    outputs = self.model.forward(torch.tensor([src]).cuda(), None)[0]
                    #print("TEST ", outputs)
                    for i in range(q_len - 1, len(src) - 1):
                        oo = outputs[i].detach().float()
                        dst = src[i + 1]
                        logit += math.log(F.softmax(oo, dim=-1)[dst])
                        _, s_index = torch.sort(oo, descending=True)
                        pred = s_index[0].item()
                        if pred != dst:
                            correct = False
                    outputs = None
                    pred = None
                logitBuf[sss] = logit
                correctBuf[sss] = correct

            res += [(logit, correct)]
            if n % 1000 == 0:
                print(f'{n // 1000}/{len(requests) // 1000}', end=' ', flush=True)
        return res

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
        results = evaluator.evaluate(
            lm=self,
            task_dict=tasks.get_task_dict(eval_tasks),
            provide_description=False,
            num_fewshot=num_fewshot,
            limit=None,
            bootstrap_iters=bootstrap_iters,
        )
        return results

#adapter = EvalHarnessAdapter(model)
#results = adapter.run_eval(
#    eval_tasks=eval_tasks,
#    bootstrap_iters=10000,
#)
#print(results['results'])