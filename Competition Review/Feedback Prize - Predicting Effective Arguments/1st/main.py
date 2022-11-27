import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoConfig, AutoModel

import os
from types import SimpleNamespace

import yaml
import multiprocessing as mp

from glob import glob

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import collections
import lightgbm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

EXP_NAME = 'efficiency-prize-v2'

N_CORES = mp.cpu_count()

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    data_folder = "test"
    df = pd.read_csv("../input/feedback-prize-effectiveness/test.csv")
    CALC_SCORE = False
else:
    data_folder = "train"
    df = pd.read_csv("../input/feedback-prize-effectiveness/train.csv")
    ids = df.essay_id.unique()
    np.random.seed(1337)
    # select the valid set
    val_ids = np.random.choice(ids, size=3000, replace=False)
    df = df[df.essay_id.isin(val_ids)]
    df = df.reset_index(drop=True)
    CALC_SCORE = True


def _read_data(essay_id):
    fname = f"../input/feedback-prize-effectiveness/{data_folder}/{essay_id}.txt"
    with open(fname) as f:
        lines = f.read()

    return lines


essay_ids = df.essay_id.unique()

pool_obj = mp.Pool(N_CORES)
results = pool_obj.map(_read_data, essay_ids)

essay_texts = dict(zip(essay_ids, results))
df["essay_text"] = df.essay_id.map(essay_texts)

cfg = yaml.safe_load(open(f"../input/{EXP_NAME}/cfg.yaml").read())
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)

cfg.architecture.cache_dir = f"../input/{EXP_NAME}/deberta-v3-large/"

tokenizer = AutoTokenizer.from_pretrained(cfg.architecture.cache_dir)

cfg._tokenizer_sep_token = tokenizer.sep_token

cfg._tokenizer_start_token_id = []
cfg._tokenizer_end_token_id = []

d_types = sorted(df.discourse_type.unique())

for t in d_types:
    tokenizer.add_tokens([f"[START_{t}]"], special_tokens=True)
    cfg._tokenizer_start_token_id.append(tokenizer.encode(f"[START_{t}]")[1])

for t in d_types:
    tokenizer.add_tokens([f"[END_{t}]"], special_tokens=True)
    cfg._tokenizer_end_token_id.append(tokenizer.encode(f"[END_{t}]")[1])

tokenizer.add_tokens([f"\n"], special_tokens=True)
cfg._tokenizer_size = len(tokenizer)




