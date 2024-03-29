import importlib
import os
NUM_GPUS = int(os.environ['RWKV_NUM_GPUS'])
USE_WANDB = (int(os.environ['USE_WANDB']) == 1)

from torch.utils.data.dataloader import DataLoader
import torch
from tqdm.auto import tqdm
import logging
import datetime
import math
from pytorch_lightning.lite import LightningLite
import gc
import torch.nn as nn
import numpy as np
import lm_evaluation

eval_tasks = ['arc_easy', 'lambada_openai', 'piqa', 'sciq']

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
if os.environ['RWKV_FLOAT_MODE'] == 'fp32':
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

class TrainerConfig:
    batch_size = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    warmup_tokens = 0
    final_tokens = 0
    epoch_save_frequency = 0
    epoch_save_path = 'wikipedia_trained/trained'
    num_workers = 0
    ctx_len = 0
    vocab_size = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

import src.model as modell
import src.model_ours as model_ours

class Trainer(LightningLite):

    def get_run_name(self):
        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        cfg = raw_model.config
        run_name = str(cfg.vocab_size) + '-' + str(cfg.ctx_len) + '-' + \
            cfg.model_type + '-' + str(cfg.n_layer) + '-' + str(cfg.n_embd)
        return run_name

    def run(self, m_cfg, train_dataset, test_dataset, config):
        self.cuda_id = int(str(self.device).strip('cuda:'))
        print('[0]')
        if config.ours:
            model = model_ours.GPT(model_ours.GPTConfig(config.vocab_size, config.ctx_len, model_type=m_cfg.model_type,
                            n_layer=m_cfg.n_layer, n_embd=m_cfg.n_embd, n_persp=config.n_persp))
        else:
            model = modell.GPT(modell.GPTConfig(config.vocab_size, config.ctx_len, model_type=m_cfg.model_type,
                                                      n_layer=m_cfg.n_layer, n_embd=m_cfg.n_embd,
                                                      n_persp=config.n_persp))
        print('[1]')
        with torch.no_grad():
            if m_cfg.LOAD_MODEL:
                print('loading', m_cfg.MODEL_NAME)
                m2 = torch.load(m_cfg.MODEL_NAME + '.pth', map_location='cpu')

                if config.ours:

                    #m2['convert.weight'] = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, (m_cfg.n_embd, config.n_persp*m_cfg.n_embd)),dtype=torch.float)
                    #                                                 , requires_grad=False)
                    m2['convert3.weight'] = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, (config.n_persp, m_cfg.n_embd)),dtype=torch.float)
                                                                     , requires_grad=False)
                    for param in m2:
                        if 'time_mix_k' in param or 'time_mix_v' in param or 'time_mix_r' in param:
                            new_params = []
                            for i in range(config.n_persp):
                               new_params.append(m2[param].clone())# + torch.tensor(np.random.normal(1, 0.01, m2[param].size()), dtype=torch.float))
                            m2[param] = nn.Parameter(torch.stack(new_params, dim=0), requires_grad=False)
                        else:
                            m2[param] = nn.Parameter(m2[param], requires_grad=False)
                model.load_state_dict(m2)
                for param in model.state_dict():
                    model.state_dict()[param].requires_grad = False
                del m2
        model.to(self.device)

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.avg_loss = -1
        self.EPOCH_BEGIN = m_cfg.EPOCH_BEGIN

        self.steps = self.EPOCH_BEGIN * (len(self.train_dataset) // (config.batch_size // NUM_GPUS))

        if self.cuda_id == 0:
            log_file = open("mylog.txt", "a")
            if USE_WANDB:
                import wandb
                cfg = model.config
                for k in config.__dict__:
                    setattr(cfg, k, config.__dict__[k])
                wandb.init(project="RWKV-LM", name=self.get_run_name() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'), config=cfg, save_code=False)

        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        model, optimizer = self.setup(model, optimizer)
        print('[3]')

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset
            data.idx_begin = self.steps * config.batch_size + 1
            data.cuda_id = self.cuda_id
            
            if config.num_workers > 0:
                loader = DataLoader(data, shuffle=False, pin_memory=False,
                                    batch_size=config.batch_size // NUM_GPUS,
                                    num_workers=config.num_workers)
            else:
                loader = DataLoader(data, shuffle=False,
                                    batch_size=config.batch_size // NUM_GPUS,
                                    num_workers=config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(
                loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)
            loader = self.setup_dataloaders(loader)
            gc.collect()
            torch.cuda.empty_cache()
            
            for it, (x, y) in pbar:
                with torch.set_grad_enabled(is_train):
                    loss = model(x, y)

                if os.environ['RWKV_DEEPSPEED'] == '0':
                    all_loss = [loss.clone()]
                else:
                    all_loss = [loss.clone() for _ in range(NUM_GPUS)]
                    torch.distributed.all_gather(all_loss, loss)

                if is_train: 
                    model.zero_grad()
                    self.backward(loss)

                    optimizer.step()

                    self.tokens += (y >= 0).sum()
                    lr_final_factor = config.lr_final / config.learning_rate
                    if self.tokens < config.warmup_tokens:
                        lr_mult = lr_final_factor + \
                            (1 - lr_final_factor) * float(self.tokens) / \
                            float(config.warmup_tokens)
                        progress = 0
                    else:
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        if progress >= 1:
                            lr_mult = lr_final_factor
                        else:
                            lr_mult = math.exp(math.log(lr_final_factor) * pow(progress, 1))
                    lr = config.learning_rate * lr_mult

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    self.lr = lr
                    self.steps += 1
                    
                    now_loss = 0
                    for gg in range(NUM_GPUS):
                        now_loss += all_loss[gg].item()
                    now_loss = now_loss / NUM_GPUS # report progress                    
                    if USE_WANDB and self.cuda_id == 0:
                        wandb.log({"loss": now_loss}, step = self.steps)

                    if self.avg_loss < 0:
                        self.avg_loss = now_loss
                    else:
                        factor = 1 / (it + 1)
                        self.avg_loss = self.avg_loss * (1.0 - factor) + now_loss * factor

                    ppl = -1

                    if self.avg_loss < 50:
                        ppl = math.exp(self.avg_loss)

                    pbar.set_description(f"miniE {epoch+1+self.EPOCH_BEGIN} s {self.steps} prog {progress*100.0:.2f}% : ppl {ppl:.6f} loss {self.avg_loss:.6f} lr {lr:e}")

        self.tokens = 0
        best_loss = 1000
        for epoch in range(8):

            run_epoch('train')
            if math.isnan(self.avg_loss):
                exit(0)

            if self.cuda_id == 0:
                log_file.write(f'{epoch+1+self.EPOCH_BEGIN} {self.avg_loss:.6f} {math.exp(self.avg_loss):.4f} {self.lr:.8f} {datetime.datetime.now()} {epoch+1} \n')
                log_file.flush()

                raw_model = self.model.module if hasattr(self.model, "module") else self.model

                torch.save(raw_model.state_dict(), self.config.epoch_save_path + '-' + str(epoch) + '.pth')