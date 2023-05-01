"""Pytorch implementation of Music Transformer with Global/Local Relative Encoding
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import AttnBlock
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map


class MusicPerceiver(nn.Module):

    def __init__(self, vocab_size, seq_len=None, n_embd=768, n_head=12, n_layer=12,
                 dropout=.1, pe_type='rotary'):
        super().__init__()

        self.n_layer = n_layer

        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        # Todo: positional encoding?

        # Cross Attn Layer
        self.cross_attn = AttnBlock(n_embd, n_head,
                                    attn_type='perceiver',
                                    pe_type=pe_type,
                                    dropout=dropout,
                                    residual=True)
        # Self Attn Layers
        self.self_attn = [AttnBlock(n_embd,
                                    n_head,
                                    attn_type='global',
                                    pe_type=pe_type,
                                    dropout=dropout,
                                    residual=True)
                          for _ in range(n_layer - 1)]

        self.layers = torch.nn.ModuleList([self.cross_attn] + self.self_attn)

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Model Parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        # Todo: multiple device
        self.device_map = (
            get_device_map(len(self.n_layer),
                           range(torch.cuda.device_count()))  # Todo: double check n_layer
            if device_map is None
            else device_map
        )
        # Todo: double check n_layer
        assert_device_map(self.device_map, self.n_layer)

        # self.transformer.parallelize(self.device_map)
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + \
            str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))

        self.wte = self.wte.to(self.first_device)
        # self.wpe = self.wpe.to(self.first_device)

        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.layers[block] = self.layers[block].to(cuda_device)

        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

        self.lm_head = self.lm_head.to(self.first_device)
        self.model_parallel = True
        return

    def deparallelize(self):
        self.transformer.deparallelize()

        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        # self.wpe = self.wpe.to("cpu")

        self.cross_attn = self.cross_attn.to("cpu")

        for index in range(len(self.n_layer)):
            self.layers[index] = self.layers[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")

        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def forward(self, x, labels=None, mask=None):
        x = self.wte(x) / math.sqrt(self.n_embd)
        # Todo: PE

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Todo: Generation

        return logits


class MusicTransormer(nn.Module):

    def __init__(self, vocab_size, seq_len=None, n_embd=768, n_head=12, n_layer=12, dropout=.1, pe=False, rel_pe=True, is_local=False):
        super().__init__()

        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, n_embd)

        self.pe = pe
        if pe:
            # positional encoding
            pass

        self.layers = torch.nn.ModuleList([AttnBlock(n_embd,
                                                     n_head,
                                                     rel_pe,
                                                     is_local)
                                           for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        # Todo: multiple device
        pass

    def forward(self, x, labels=None, mask=None):
        x = self.wte(x) / math.sqrt(self.n_embd)

        if self.pe:
            pos = self.wpe(x)
            x = x + pos

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        return x
