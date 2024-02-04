import os
import logging, types
from src.utils import Dataset
import torch
import numpy as np
from src.binidx import MMapIndexedDataset
from datasets import load_dataset, load_from_disk

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

if False: 
    import src.utils
    src.utils.set_seed(42)

EXPRESS_PILE_MODE = False 

EXPRESS_PILE_MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'
EXPRESS_PILE_MODEL_TYPE = 'RWKV-4-Pile-169M'

device = 'cuda'
ours = False

n_persp = 4


from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

tokenized = False
if tokenized:
    datafile = load_from_disk("../data/final_saved")
else:
    datafile = load_dataset("wikipedia", "20220301.en", cache_dir="/home/gigi/hdd/RNNWins/data")

print(datafile, ' ', len(datafile))

datafile_encoding = 'huggingface'

if datafile_encoding == 'huggingface':

    if not tokenized:

        def tokenization(example):
            example['text'] = tokenizer(example['text']).input_ids
            return example

        print(datafile)

        dataset = datafile['train'].remove_columns(['id', 'url', 'title']).map(tokenization, batched=True)

        dataset.save_to_disk("../data/final_saved")

        with open('data/wikipedia.npy', 'wb') as f:
            np.save(f, dataset['text'])

if EXPRESS_PILE_MODE:
    datafile = 'train.npy'
    datafile_encoding = 'numpy'

os.environ['VOCAB_SIZE'] = '50277'

os.environ['RWKV_NUM_GPUS'] = '1'

os.environ['RWKV_FLOAT_MODE'] = 'bf16'

os.environ['RWKV_DEEPSPEED'] = '1' 

if int(os.environ['RWKV_NUM_GPUS']) == 1:
    os.environ['RWKV_DEEPSPEED'] = '0' 

os.environ['USE_WANDB'] = '0' 

EPOCH_BEGIN = 0
LOAD_MODEL = False 

n_layer = 12
n_embd = 768
ctx_len = 1024

model_type = 'RWKV'

LOAD_MODEL = True
if EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-169M':
    n_layer = 12
    n_embd = 768
    ctx_len = 1024
elif EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-430M':
    n_layer = 24
    n_embd = 1024
    ctx_len = 1024
elif EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-1B5':
    n_layer = 24
    n_embd = 2048
    ctx_len = 1024

batch_size = 2 * int(os.environ['RWKV_NUM_GPUS'])
assert (batch_size % int(os.environ['RWKV_NUM_GPUS']) == 0)

lr_init = 3e-5
lr_final = 1e-5

n_epoch = 8
epoch_length_fixed = (16000 // batch_size) * batch_size 

epoch_save_frequency = 10
epoch_save_path = 'wikipedia_trained/trained'

if EXPRESS_PILE_MODE:
    lr_init = 1e-5
    lr_final = 1e-5
    n_epoch = 100000

if LOAD_MODEL and EPOCH_BEGIN > 0:
    warmup_tokens = 200 * ctx_len * batch_size // int(os.environ['RWKV_NUM_GPUS'])
else:
    warmup_tokens = 0

betas = (0.9, 0.99) 
eps = 1e-8

num_workers = 1 

NUM_GPUS = int(os.environ['RWKV_NUM_GPUS'])
os.environ['RWKV_LOAD_MODEL'] = str(LOAD_MODEL)

betas = (0.9, 0.999)
MODEL_NAME = 'weights/' + EXPRESS_PILE_MODEL_NAME

torch.backends.cudnn.benchmark = True
if os.environ['RWKV_FLOAT_MODE'] == 'fp32':
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

print(f'loading {datafile_encoding} data... ' + str(datafile))
if datafile_encoding == 'binidx':
    train_dataset = Dataset(MMapIndexedDataset(datafile), ctx_len, epoch_length_fixed)
elif datafile_encoding == 'numpy':
    train_dataset = Dataset(np.load(datafile).astype('int'), ctx_len, epoch_length_fixed)
elif datafile_encoding == 'huggingface':
    train_dataset = Dataset(datafile.with_format("torch", device=device), ctx_len, epoch_length_fixed, tokenizer, hugging_face=True)
else:
    train_dataset = Dataset(open(datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn', force=True)

    from src.trainer import Trainer, TrainerConfig

    print('\nmodel', model_type, os.environ['RWKV_FLOAT_MODE'], 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, '\n')

    tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size, ctx_len=ctx_len, vocab_size=int(os.environ['VOCAB_SIZE']),
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, n_persp=n_persp, ours=ours,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    m_cfg = types.SimpleNamespace()
    m_cfg.model_type = model_type
    m_cfg.n_layer = n_layer
    m_cfg.n_embd = n_embd
    m_cfg.EPOCH_BEGIN = EPOCH_BEGIN
    m_cfg.LOAD_MODEL = LOAD_MODEL
    m_cfg.MODEL_NAME = MODEL_NAME

    if os.environ['RWKV_DEEPSPEED'] == '0':
        if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision=16)            
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision='bf16')
        elif '32' in os.environ['RWKV_FLOAT_MODE']:
            trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision=32)
    else:
        from pytorch_lightning.strategies import DeepSpeedStrategy
        
        DEEPSPEED_CFG = {
            "zero_allow_untested_optimizer":True,
            "zero_optimization":{
                "stage":2,
                "contiguous_gradients":True,
                "overlap_comm":True,
                "allgather_partitions":True,
                "reduce_scatter":True,
                "allgather_bucket_size":200000000,
                "reduce_bucket_size":200000000,
                "sub_group_size":1000000000000
            },
            "activation_checkpointing":{
                "partition_activations":False,
                "cpu_checkpointing":False,
                "contiguous_memory_optimization":False,
                "synchronize_checkpoint_boundary":False
            },
            "aio":{
                "block_size":1048576,
                "queue_depth":8,
                "single_submit":False,
                "overlap_events":True,
                "thread_count":1
            },
            "gradient_clipping": 1.0,
            "gradient_accumulation_steps": 1,
        }
        if NUM_GPUS == 1:
            DEEPSPEED_CFG['zero_optimization'] = {
                "stage":1, # saves some VRAM
                "contiguous_gradients":False,
                "overlap_comm":False,
                "allgather_partitions":False,
                "reduce_scatter":False,
                "allgather_bucket_size":200000000,
                "reduce_bucket_size":200000000,
                "sub_group_size":1000000000000
            }

        if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            DEEPSPEED_CFG["fp16"] = {
                "fp16": True,
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 12,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
            trainer = Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=NUM_GPUS, accelerator="gpu", precision=16)
            
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            DEEPSPEED_CFG["bf16"] = {
                "enabled": True
            }
            trainer = Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=NUM_GPUS, accelerator="gpu", precision='bf16')

        elif '32' in os.environ['RWKV_FLOAT_MODE']:
            trainer = Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=NUM_GPUS, accelerator="gpu", precision=32)

        print(trainer._strategy.config)
    
    trainer.run(m_cfg, train_dataset, None, tconf)
