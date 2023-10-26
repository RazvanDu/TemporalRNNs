########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types
import copy
import torch
import math, os
from torch.nn import functional as F
import torch.nn as nn
import numpy
import torch.nn.init as init

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

DEBUG_TIME = False  # True False - show trained time-coeffs

class RWKV_RNN(nn.Module):  # this is running in FP32 at this moment
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len, load):
        super().__init__()
        self.clear()
        self.RUN_DEVICE = RUN_DEVICE
        self.model_type = model_type
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.ctx_len = ctx_len
        self.number_persp = 8
        self.exp_persp = 1

        self.classifier = nn.Linear(self.number_persp * self.n_embd, self.n_embd).cuda()

        self.w = types.SimpleNamespace()

        # w = torch.load('weights/' + MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE))
        w = torch.load('weights/' + MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE))

        self.target = []

        if load is not None and load != False:
            self.loaded = torch.load('saves/' + load, map_location=torch.device(RUN_DEVICE))
            print("Loading trained weights...")

        for x in w.keys():

            w[x] = w[x].float()

            if '.time_' in x:
                w[x] = w[x].squeeze()

            if '.time_mix' in x:# or '.receptance' in x:

                a = nn.Parameter(w[x], requires_grad=False)
                # print("QQ ", str(w[x]))
                # print("WW ", str(self.a))

                w[x] = []
                replaced = x.replace(".", "")
                w[x].append(a)

                for i in range(1, self.number_persp):
                    w[x].append(w[x][i - 1] * self.exp_persp)

                for i in range(1, self.number_persp):
                    #xavier_matrix = torch.empty_like(w[x][i], requires_grad=False)
                    #init.xavier_uniform_(xavier_matrix)
                    noise = torch.tensor(numpy.random.normal(0, 1, w[x][i].size()), dtype=torch.float).cuda()
                    w[x][i] = nn.Parameter(w[x][i] + noise/5, requires_grad=True)

                for i in range(self.number_persp):
                    if load:
                        w[x][i] = nn.Parameter(self.loaded[replaced + str(i)].float(), requires_grad=True)
                    self.register_parameter(replaced + str(i), w[x][i])

                self.examples = []

                for i in range(int(self.number_persp/2)):
                    self.examples.append(w[x][i])

                #if '.time_mix' in x:
                #    self.example1 = w[x][1]
                #if '.receptance' in x:
                #    self.example2 = w[x][1]
                #self.example4 = w[x][3]
            if '.time_decay' in x:
                w[x] = -torch.exp(w[x])
            if DEBUG_TIME and '.time_' in x:
                print(x, w[x].squeeze().cpu().numpy())

            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i + 1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

            # TODO: MAKE ALL THE NETWORK TRAIN_ABLE

            # if '.receptance' not in x:
            # eplaced = x.replace(".", "")
            # w[x] = nn.Parameter(w[x], requires_grad=True)
            # self.register_parameter(replaced, w[x])

        # self.target2 = nn.ParameterList(self.target)
        # self.recpt =w.receptance.weight       self.clear()

    def clear(self):
        self.xx = {}
        self.aa = {}
        self.bb = {}
        self.pp = {}
        self.hk = None

    def save(self, target):
        target.xx = copy.deepcopy(self.xx)
        target.aa = copy.deepcopy(self.aa)
        target.bb = copy.deepcopy(self.bb)
        target.pp = copy.deepcopy(self.pp)
        target.hk = copy.deepcopy(self.hk)

    def load(self, target):
        self.xx = copy.deepcopy(target.xx)
        self.aa = copy.deepcopy(target.aa)
        self.bb = copy.deepcopy(target.bb)
        self.pp = copy.deepcopy(target.pp)
        self.hk = copy.deepcopy(target.hk)

    def dettachh(self):

        for name in self.xx.keys():
            self.xx[name] = torch.stack([torch.stack(self.xx[name][j]) for j in range(len(self.xx[name]))]).detach()
        for name in self.aa.keys():
            self.aa[name] = torch.stack([torch.stack(self.aa[name][j]) for j in range(len(self.aa[name]))]).detach()
        for name in self.bb.keys():
            self.bb[name] = torch.stack([torch.stack(self.bb[name][j]) for j in range(len(self.bb[name]))]).detach()
        for name in self.pp.keys():
            self.pp[name] = torch.stack([torch.stack(self.pp[name][j]) for j in range(len(self.pp[name]))]).detach()

    def initt(self):

        for name in self.xx.keys():
            self.xx[name] = [[value_n for value_n in value] for value in self.xx[name]]
        for name in self.aa.keys():
            self.aa[name] = [[value_n for value_n in value] for value in self.aa[name]]
        for name in self.bb.keys():
            self.bb[name] = [[value_n for value_n in value] for value in self.bb[name]]
        for name in self.pp.keys():
            self.pp[name] = [[value_n for value_n in value] for value in self.pp[name]]
        # TODO: DETACH HK?

    def LN(self, xx, w):

        result = []

        for i in range(self.number_persp):
            result.append(F.layer_norm(xx[i].clone(), (self.n_embd,), weight=w.weight, bias=w.bias))

        return result

    def FF(self, xx, w, name):

        if name not in self.xx:
            self.xx[name] = [torch.zeros(self.n_embd, device=self.RUN_DEVICE) for _ in range(self.number_persp)]

        result = []

        for i in range(self.number_persp):

            xk = xx[i] * w.time_mix_k[i] + self.xx[name][i] * (1 - w.time_mix_k[i])
            xr = xx[i] * w.time_mix_r[i] + self.xx[name][i] * (1 - w.time_mix_r[i])

            r = torch.sigmoid(w.receptance.weight @ xr)

            k = torch.square(torch.relu(w.key.weight @ xk))
            kv = w.value.weight @ k

            result.append(r * kv)

        self.xx[name] = xx

        return result

    def SA(self, xx, w, name):

        if name not in self.xx:
            self.xx[name] = [torch.zeros(self.n_embd, device=self.RUN_DEVICE) for _ in range(self.number_persp)]
            self.aa[name] = [torch.zeros(self.n_embd, device=self.RUN_DEVICE) for _ in range(self.number_persp)]
            self.bb[name] = [torch.zeros(self.n_embd, device=self.RUN_DEVICE) for _ in range(self.number_persp)]
            self.pp[name] = [torch.zeros(self.n_embd, device=self.RUN_DEVICE) - 1e30 for _ in range(self.number_persp)]

        result = []

        for i in range(self.number_persp):
            xk = xx[i] * w.time_mix_k[i] + self.xx[name][i] * (1 - w.time_mix_k[i])
            xv = xx[i] * w.time_mix_v[i] + self.xx[name][i] * (1 - w.time_mix_v[i])
            xr = xx[i] * w.time_mix_r[i] + self.xx[name][i] * (1 - w.time_mix_r[i])

            r = torch.sigmoid(w.receptance.weight @ xr)

            self.recept = w.key.weight

            k = w.key.weight @ xk
            v = w.value.weight @ xv

            pp = self.pp[name][i]
            aa = self.aa[name][i]
            bb = self.bb[name][i]

            ww = w.time_first + k
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            a = e1 * aa + e2 * v
            b = e1 * bb + e2
            ww = pp + w.time_decay
            p = torch.maximum(ww, k)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(k - p)
            self.aa[name][i] = (e1 * aa + e2 * v)
            self.bb[name][i] = (e1 * bb + e2)
            self.pp[name][i] = (p)
            rwkv = r * a / b

            result.append(w.output.weight @ rwkv)

        self.xx[name] = xx

        return result

    def forward(self, ctx):

        # self.initt()

        w = self.w
        x = w.emb.weight[ctx[-1]]

        copyy = x
        x = []

        for i in range(self.number_persp):
            x.append(copyy)

        x = torch.stack(x)

        for i in range(self.n_layer):
            if i == 0:
                temp_x = self.LN(x, w.blocks[i].ln0)

                for j in range(self.number_persp):
                    x[j] = temp_x[j]
            if i == 0 and self.model_type == 'RWKV-ffnPre':
                temp_x = self.FF(self.LN(x, w.blocks[i].ln1), w.blocks[i].ffnPre, f'ffnPre.{i}')

                for j in range(self.number_persp):
                    x[j] += temp_x[j]
            else:
                temp_x = self.SA(self.LN(x, w.blocks[i].ln1), w.blocks[i].att, f'att.{i}')

                for j in range(self.number_persp):
                    x[j] += temp_x[j]

            temp_ff = self.FF(self.LN(x, w.blocks[i].ln2), w.blocks[i].ffn, f'ffn.{i}')

            for j in range(self.number_persp):
                x[j] += temp_ff[j]

        x = self.LN(x, w.ln_out)

        if RWKV_HEAD_QK_DIM > 0:
            if self.hk == None:
                self.hk = (w.head_k.weight @ x).unsqueeze(0)
            else:
                self.hk = torch.cat(
                    [self.hk, (w.head_k.weight @ x).unsqueeze(0)], dim=0)
            if self.hk.shape[0] > self.ctx_len:
                self.hk = self.hk[-self.ctx_len:, :]

            q = w.head_q.weight @ x

            x = w.head.weight @ x
            x = x.cpu().numpy().tolist()

            c = (self.hk @ q) / RWKV_HEAD_QK_DIM
            for i in range(len(c)):
                x[ctx[i]] += c[i]
        else:

            sum = 0

            for i in range(self.number_persp):
                sum += x[i] / self.number_persp
            x = w.head.weight @ sum

            #x = self.classifier()

        # self.dettachh()

        with torch.no_grad():

            for name in self.xx:
                for key in range(len(self.xx[name])):
                    self.xx[name][key] = self.xx[name][key].detach()
            for name in self.aa:
                for key in range(len(self.aa[name])):
                    self.aa[name][key] = self.aa[name][key].detach()
            for name in self.bb:
                for key in range(len(self.bb[name])):
                  self.bb[name][key] = self.bb[name][key].detach()
            for name in self.pp:
                for key in range(len(self.pp[name])):
                    self.pp[name][key] = self.pp[name][key].detach()

        return x

# class GREBE_RNN(nn.Module): # this is running in FP32 at this moment
#     def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len, load):
#         super().__init__()
#         self.clear()
#         self.RUN_DEVICE = RUN_DEVICE
#         self.model_type = model_type
#         self.n_layer = n_layer
#         self.n_embd = n_embd
#         self.ctx_len = ctx_len
#         self.number_persp = 16
#         self.exp_persp = 1
#
#         #self.linear_1 = nn.Linear(self.n_embd, self.n_embd, device=RUN_DEVICE)
#         #self.linear_2 = nn.Linear(self.n_embd * self.number_persp, self.n_embd, device=RUN_DEVICE)
#         #self.linear_3 = nn.Linear(self.n_embd, self.n_embd, device=RUN_DEVICE)
#
#         self.w = types.SimpleNamespace()
#
#         #w = torch.load('weights/' + MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE))
#         w = torch.load('weights/' + MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE))
#
#         self.target = []
#
#         if load:
#             self.loaded = torch.load('saves/' + 'best_hopefully_16persp', map_location=torch.device(RUN_DEVICE))
#             print("Loading trained weights...")
#
#         for x in w.keys():
#
#             w[x] = w[x].float()
#
#             if '.receptance' in x:
#
#                 self.a = nn.Parameter(w[x], requires_grad=False)
#                 #print("QQ ", str(w[x]))
#                 #print("WW ", str(self.a))
#
#                 w[x] = []
#                 replaced = x.replace(".", "")
#                 w[x].append(self.a)
#
#                 for i in range(1, self.number_persp):
#                     w[x].append(nn.Parameter(w[x][i-1] * self.exp_persp, requires_grad=False))
#
#                 for i in range(self.number_persp):
#                     if load:
#                         w[x][i] = nn.Parameter(self.loaded[replaced + str(i)].float(), requires_grad=False)
#                     self.register_parameter(replaced + str(i), w[x][i])
#
#                 self.example1 = w[x][0]
#                 #self.example2 = w[x][1]
#                 #self.example3 = w[x][2]
#                 #self.example4 = w[x][3]
#
#             if '.time_' in x:
#                 w[x] = w[x].squeeze()
#             if '.time_decay' in x:
#                 w[x] = -torch.exp(w[x])
#             if DEBUG_TIME and '.time_' in x:
#                 print(x, w[x].squeeze().cpu().numpy())
#
#             xx = x.split('.')
#             here = self.w
#             for i in range(len(xx)):
#                 if xx[i].isdigit():
#                     ii = int(xx[i])
#                     if ii not in here:
#                         here[ii] = types.SimpleNamespace()
#                     here = here[ii]
#                 else:
#                     if i == len(xx) - 1:
#                         setattr(here, xx[i], w[x])
#                     elif not hasattr(here, xx[i]):
#                         if xx[i+1].isdigit():
#                             setattr(here, xx[i], {})
#                         else:
#                             setattr(here, xx[i], types.SimpleNamespace())
#                     here = getattr(here, xx[i])
#
#         #self.target2 = nn.ParameterList(self.target)
#         #self.recpt =w.receptance.weight       self.clear()
#
#     def clear(self):
#         self.xx = {}
#         self.aa = {}
#         self.bb = {}
#         self.pp = {}
#         self.hk = None
#
#     def save(self, target):
#         target.xx = copy.deepcopy(self.xx)
#         target.aa = copy.deepcopy(self.aa)
#         target.bb = copy.deepcopy(self.bb)
#         target.pp = copy.deepcopy(self.pp)
#         target.hk = copy.deepcopy(self.hk)
#
#     def load(self, target):
#         self.xx = copy.deepcopy(target.xx)
#         self.aa = copy.deepcopy(target.aa)
#         self.bb = copy.deepcopy(target.bb)
#         self.pp = copy.deepcopy(target.pp)
#         self.hk = copy.deepcopy(target.hk)
#
#     def dettachh(self):
#
#         for name in self.xx.keys():
#             self.xx[name] = self.xx[name].detach()
#         for name in self.aa.keys():
#             self.aa[name] = self.aa[name].detach()
#         for name in self.bb.keys():
#             self.bb[name] = self.bb[name].detach()
#         for name in self.pp.keys():
#             self.pp[name] = self.pp[name].detach()
#
#     def initt(self):
#
#         print()
#
#         #for name in self.xx.keys():
#         #    self.xx[name] = [[value_n.clone() for value_n in value] for value in self.xx[name]]
#         #for name in self.aa.keys():
#         #    self.aa[name] = [[value_n.clone() for value_n in value] for value in self.aa[name]]
#         #for name in self.bb.keys():
#         #    self.bb[name] = [[value_n.clone() for value_n in value] for value in self.bb[name]]
#         #for name in self.pp.keys():
#         #    self.pp[name] = [[value_n.clone() for value_n in value] for value in self.pp[name]]
#         # TODO: DETACH HK?
#
#     def LN(self, xx, w):
#
#         result = torch.zeros(self.number_persp, self.n_embd, device=self.RUN_DEVICE)
#
#         for i in range(self.number_persp):
#             result.[F.layer_norm(xx[i].clone(), (self.n_embd,), weight=w.weight, bias=w.bias))
#
#         return result
#
#     def FF(self, xx, w, name):
#
#         if name not in self.xx:
#             self.xx[name] = torch.zeros(2, self.number_persp, self.n_embd, device=self.RUN_DEVICE)
#
#         self.xx[name][0] = self.xx[name][1]
#         self.xx[name][1] = xx
#
#         #self.xx[name].append(xx)
#
#         #if len(self.xx[name]) > 2:
#         #    self.xx[name].pop(0)
#
#         result = []
#
#         for i in range(self.number_persp):
#
#             xk = xx[i] * w.time_mix_k + self.xx[name][0][i] * (1 - w.time_mix_k)
#             xr = xx[i] * w.time_mix_r + self.xx[name][0][i] * (1 - w.time_mix_r)
#
#             r = torch.sigmoid(w.receptance.weight[i].clone() @ xr)
#
#             k = torch.square(torch.relu(w.key.weight @ xk))
#             kv = w.value.weight @ k
#
#             result.append(r * kv)
#
#         return result
#
#     def SA(self, xx, w, name):
#
#         if name not in self.xx:
#             self.xx[name] = torch.zeros(2, self.number_persp, self.n_embd, device=self.RUN_DEVICE)
#             self.aa[name] = torch.zeros(2, self.number_persp, self.n_embd, device=self.RUN_DEVICE)
#             self.bb[name] = torch.zeros(2, self.number_persp, self.n_embd, device=self.RUN_DEVICE)
#             self.pp[name] = torch.zeros(2, self.number_persp, self.n_embd, device=self.RUN_DEVICE) - 1e30
#
#         self.xx[name][0] = self.xx[name][1]
#         self.aa[name][0] = self.aa[name][1]
#         self.bb[name][0] = self.bb[name][1]
#         self.pp[name][0] = self.pp[name][1]
#
#         self.xx[name][1] = xx
#         #self.aa[name].append([torch.zeros(self.n_embd, device=self.RUN_DEVICE) for _ in range(self.number_persp)])
#         #self.bb[name].append([torch.zeros(self.n_embd, device=self.RUN_DEVICE) for _ in range(self.number_persp)])
#         #self.pp[name].append([torch.zeros(self.n_embd, device=self.RUN_DEVICE) - 1e30 for _ in range(self.number_persp)])
#
#         #if len(self.xx[name]) > 2:
#         #    self.xx[name].pop(0)
#         #    self.aa[name].pop(0)
#         #    self.bb[name].pop(0)
#         #    self.pp[name].pop(0)
#
#         result =torch.zeros(self.number_persp, self.n_embd, device=self.RUN_DEVICE)
#
#         for i in range(self.number_persp):
#
#             xk = xx[i] * w.time_mix_k + self.xx[name][0][i] * (1 - w.time_mix_k)
#             xv = xx[i] * w.time_mix_v + self.xx[name][0][i] * (1 - w.time_mix_v)
#             xr = xx[i] * w.time_mix_r + self.xx[name][0][i] * (1 - w.time_mix_r)
#
#             r = torch.sigmoid(w.receptance.weight[i].clone() @ xr)
#
#             k = w.key.weight @ xk
#             v = w.value.weight @ xv
#
#             pp = self.pp[name][0][i]
#             aa = self.aa[name][0][i]
#             bb = self.bb[name][0][i]
#
#             ww = w.time_first + k
#             p = torch.maximum(pp, ww)
#             e1 = torch.exp(pp - p)
#             e2 = torch.exp(ww - p)
#             a = e1 * aa + e2 * v
#             b = e1 * bb + e2
#             ww = pp + w.time_decay
#             p = torch.maximum(ww, k)
#             e1 = torch.exp(ww - p)
#             e2 = torch.exp(k - p)
#             self.aa[name][1][i] = (e1 * aa + e2 * v)
#             self.bb[name][1][i] = (e1 * bb + e2)
#             self.pp[name][1][i] = (p)
#             rwkv = r * a / b
#
#             result.append(w.output.weight @ rwkv)
#
#         return result
#
#     def forward(self, ctx):
#
#         self.initt()
#
#         w = self.w
#         x = w.emb.weight[ctx[-1]]
#
#         copyy = x
#         x = torch.zeros(self.number_persp, self.n_embd, device=self.RUN_DEVICE)
#
#         for i in range(self.number_persp):
#             x[i] = copyy.clone()
#
#         for i in range(self.n_layer):
#             if i == 0:
#                 temp_x = self.LN(x, w.blocks[i].ln0)
#
#                 for j in range(self.number_persp):
#                     x[j] = temp_x[j]
#             if i == 0 and self.model_type == 'RWKV-ffnPre':
#                 temp_x = self.FF(self.LN(x, w.blocks[i].ln1), w.blocks[i].ffnPre, f'ffnPre.{i}')
#
#                 for j in range(self.number_persp):
#                     x[j] += temp_x[j]
#             else:
#                 temp_x = self.SA(self.LN(x, w.blocks[i].ln1), w.blocks[i].att, f'att.{i}')
#
#                 for j in range(self.number_persp):
#                     x[j] += temp_x[j]
#
#             temp_ff = self.FF(self.LN(x, w.blocks[i].ln2), w.blocks[i].ffn, f'ffn.{i}')
#
#             for j in range(self.number_persp):
#                 x[j] += temp_ff[j]
#
#         x = self.LN(x, w.ln_out)
#
#         if RWKV_HEAD_QK_DIM > 0:
#             if self.hk == None:
#                 self.hk = (w.head_k.weight @ x).unsqueeze(0)
#             else:
#                 self.hk = torch.cat(
#                     [self.hk, (w.head_k.weight @ x).unsqueeze(0)], dim=0)
#             if self.hk.shape[0] > self.ctx_len:
#                 self.hk = self.hk[-self.ctx_len:, :]
#
#             q = w.head_q.weight @ x
#
#             x = w.head.weight @ x
#             x = x.cpu().numpy().tolist()
#
#             c = (self.hk @ q) / RWKV_HEAD_QK_DIM
#             for i in range(len(c)):
#                 x[ctx[i]] += c[i]
#         else:
#
#             sum = 0
#
#             for i in range(self.number_persp):
#                 sum += x[i] / self.number_persp
#             x = w.head.weight @ sum
#
#         self.dettachh()
#
#         return x
