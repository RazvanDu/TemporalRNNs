########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math, os
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from deepspeed.ops.adam import FusedAdam
except:
    pass # some poor windows users cant install deepspeed

logger = logging.getLogger(__name__)

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss
    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = 1024 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            w = -torch.exp(w.float().contiguous())
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return y
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return y.half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return y.bfloat16()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        else:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return (None, None, None, gw, gu, gk, gv)
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(model, args):  # fancy initialization of all lin & emb layer in the model
    print("\n[--> first run, init model params (very slow for large models) <--]")
    print("[so you shall only do it for 1 single GPU and save the checkpt and load it when using multiple GPU]\n")

    for mm in model.modules():
        if "RecursiveScriptModule" in str(type(mm)):
            if mm.original_name not in ["Linear"]:
                continue
            ww = None
            for name, param in mm.named_parameters():
                if name == "weight":
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            ww = m.weight
        with torch.no_grad():
            name = "[unknown weight]"
            for name, parameter in model.named_parameters():  # find the name of the weight
                if id(ww) == id(parameter):
                    break

            shape = ww.shape
            gain = 1.0
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:  # token emb?
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:  # final projection?
                    scale = 0.5

            if hasattr(m, "scale_init"):
                scale = m.scale_init

            # print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {name}")

            gain *= scale
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                # zero init is great for some RWKV matrices
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)


class RWKV_TimeMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.n_persp = config.n_persp
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        with torch.no_grad(): # fancy init
            ratio_0_to_1 = (layer_id / (config.n_layer - 1)) # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer)) # 1 to ~0
            
            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
            
            # fancy time_mix
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_k = nn.Parameter(torch.stack([torch.pow(x, ratio_1_to_almost0) for _ in range(config.n_persp)], dim=0))
            self.time_mix_v = nn.Parameter(torch.stack([torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1 for _ in range(config.n_persp)], dim=0))
            self.time_mix_r = nn.Parameter(torch.stack([torch.pow(x, 0.5 * ratio_1_to_almost0) for _ in range(config.n_persp)], dim=0))
            
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    @torch.jit.script_method
    def jit_func(self, x, i: int):

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k[i] + xx * (1 - self.time_mix_k[i])
        xv = x * self.time_mix_v[i] + xx * (1 - self.time_mix_v[i])
        xr = x * self.time_mix_r[i] + xx * (1 - self.time_mix_r[i])

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x: list[torch.Tensor]):

        output = []

        for i in range(self.n_persp):

            B, T, C = x[i].size() # x = (Batch,Time,Channel)

            sr, k, v = self.jit_func(x[i], i)

            rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
            output.append(self.output(rwkv))

        return output


class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.n_persp = config.n_persp
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad(): # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer)) # 1 to ~0

            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd

            self.time_mix_k = nn.Parameter(torch.stack([torch.pow(x, ratio_1_to_almost0) for _ in range(config.n_persp)], dim=0))
            self.time_mix_r = nn.Parameter(torch.stack([torch.pow(x, ratio_1_to_almost0) for _ in range(config.n_persp)], dim=0))

        hidden_sz = 4 * config.n_embd
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    @torch.jit.script_method
    def forward(self, x: list[torch.Tensor]):

        output = []

        for i in range(self.n_persp):

            xx = self.time_shift(x[i])
            xk = x[i] * self.time_mix_k[i] + xx * (1 - self.time_mix_k[i])
            xr = x[i] * self.time_mix_r[i] + xx * (1 - self.time_mix_r[i])

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            kv = self.value(k)

            output.append(torch.sigmoid(self.receptance(xr)) * kv)

        return output

########################################################################################################
# The GPT Model with our blocks
########################################################################################################


class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, 0)
        else:
            self.att = RWKV_TimeMix(config, layer_id)

        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x: list[torch.Tensor]):
        if self.layer_id == 0:
            for i in range(self.config.n_persp):
                x[i] = self.ln0(x[i])
        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            bef = [x[i].clone() for i in range(len(x))]
            for i in range(self.config.n_persp):
                x[i] = self.ln1(x[i])
            temp = self.ffnPre(x)
            for i in range(self.config.n_persp):
                x[i] = bef[i] + temp[i]  # better in some cases
        else:
            bef = [x[i].clone() for i in range(len(x))]
            for i in range(self.config.n_persp):
                x[i] = self.ln1(x[i])
            temp = self.att(x)
            for i in range(self.config.n_persp):
                x[i] = bef[i] + temp[i]
        bef = [x[i].clone() for i in range(len(x))]
        for i in range(self.config.n_persp):
            x[i] = self.ln2(x[i])
        temp = self.ffn(x)
        for i in range(self.config.n_persp):
            x[i] = bef[i] + temp[i]
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.convert = nn.Linear(config.n_embd*config.n_persp, config.n_embd, bias=False)
        self.convert2 = nn.Linear(config.n_embd, config.n_persp, bias=False)
        self.convert3 = nn.Linear(config.n_embd*config.n_persp, config.n_persp, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", torch.tril(
                torch.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len

        try:
            if os.environ['RWKV_LOAD_MODEL'] == str(False):
                RWKV_Init(self, config) 
        except:
            pass

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        no_decay = set()

        #print("CONFIGUREEE")

        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if 'time_mix' in fpn or 'convert' in fpn:
                    #print(fpn)
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        try:
            optimizer = FusedAdam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        except:
            print('\n\nDeepSpeed not found. Using torch optimizer instead (probably slower)\n\n')
            optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)

        return optimizer

    def forward(self, idx, targets=None):
        idx = idx.to(self.emb.weight.device)

        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        temp = self.emb(idx)
        #print("XXX " , temp.size())
        x = [temp for _ in range(self.config.n_persp)]
        x = self.blocks(x)
        for i in range(self.config.n_persp):
            x[i] = self.ln_out(x[i])

        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size)
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).half()
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).bfloat16()

            x = self.head(x) + c
        else:
            #print("W ", x[0].size())
            #print("T ", torch.cat(x, dim=2).size())
            #print("Q ", torch.cat(x, dim=0).size())
            #print("T ", self.convert(torch.cat(x, dim=2)))
            #weightss = []
            #for i in range(self.config.n_persp):
            #    weightss.append(self.convert(x[i]))
            #weightss = self.softmax(torch.cat(weightss, dim=2))
            #weightss = self.softmax(self.convert2(self.convert(torch.cat(x, dim=2))))
            weightss = self.softmax(self.convert3(torch.cat(x, dim=2)))
            #print("E ", self.convert(torch.cat(x, dim=2)).size())
            print("R ", self.convert.weight)
            #print("R2 ", self.convert2.weight)
            for i in range(self.config.n_persp):
                x[i] = self.head(x[i])
            #print("Q ", x.size())

        print(x[0].size())
        print(weightss.size())

        #for B in len(x[0]):

        x = [x[i] * weightss[..., i].unsqueeze(-1) for i in range(self.config.n_persp)]
        x = sum(x)
        #x = torch.mean(torch.stack(x), dim=0)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.to(x.device).view(-1))

        return L2Wrap.apply(loss, x)
