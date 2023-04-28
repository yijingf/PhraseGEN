"""Pytorch implementation of Music Transformer with Global/Local Relative Encoding
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import DecoderLayer


class MusicPerceiver(nn.Module):

    def __init__(self, vocab_size, seq_len=None, n_embd=768, n_head=12, n_layer=12, dropout=.1, pe=False, rel_pe=True, is_local=False):
        super().__init__()

        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, n_embd)

        self.pe = pe
        if pe:
            # positional encoding
            pass

        self.perceiver = DecoderLayer(n_embd, n_head, attn_type='perceiver')
        self.layers = torch.nn.ModuleList([copy.deepcopy(DecoderLayer(n_embd,
                                                                      n_head,
                                                                      attn_type='global', rel_pe=False))
                                           for _ in range(n_layer - 1)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        # Todo: multiple device
        pass

    def forward(self, x, labels=None, mask=None):
        x = self.wte(x) / math.sqrt(self.n_embd)

        # Todo: PE

        x = self.perceiver(x)
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

        self.layers = torch.nn.ModuleList([copy.deepcopy(DecoderLayer(n_embd,
                                                                      n_head,
                                                                      rel_pe,
                                                                      is_local))
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
