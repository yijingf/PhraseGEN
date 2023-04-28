import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, n_embd, n_inter, dropout=.1):
        # hugging face use conv1d
        self.fc = nn.Linear(n_embd, n_inter, bias=False)
        self.proj = nn.Linear(n_inter, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)

        x = self.proj(x)
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, n_fc=3072, max_len=1024, rel_pe=False, attn_type='global', dropout=.1):

        if attn_type == 'local':
            self.attn = LocalAttention()
        elif attn_type == 'global':
            self.attn = GlobalAttention(n_embd, n_head, max_len, rel_pe)
        elif attn_type == 'perceiver':
            self.attn = PerceiverAttention(n_embd, n_head, max_len, rel_pe)
        else:
            raise ValueError("Unknown type of attention Module")

        self.ln_1 = nn.LayerNorm(n_embd)
        self.resid_dropout = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp_block = MLPBlock(n_embd, n_fc, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        x = self.ln_1(x)
        attn = self.attn(x, mask)
        x = self.resid_dropout(x + attn)

        x = self.ln_2(x)
        x = self.mlp_block(x)

        return self.dropout(x)


class PerceiverAttention(nn.Module):
    def __init__(self, max_len, con_len, n_embd, n_head, dropout=.1):
        super().__init__()
        d_head, remainder = divmod(n_embd, n_head)
        if remainder:
            raise ValueError("incompatible `n_embd` and `n_head`")

        self.max_len = max_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x, block_len, mask=None):
        # Todo: checkout generation mode
        batch_size, seq_len, _ = x.shape

        x_kv = self.ln(x)
        x_q = self.ln(x[:, -block_len:, :, ])

        k_t = self.key(x_kv).reshape(batch_size, seq_len,
                                     self.n_head, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, n_head, d_head, seq_len)
        v = self.value(x_kv).reshape(batch_size, seq_len,
                                     self.n_head, -1).transpose(1, 2)
        # shape = (batch_size, n_head, seq_len, d_head)
        q = self.query(x_q).reshape(batch_size, block_len,
                                    self.n_head, -1).transpose(1, 2)
        # shape = (batch_size, n_head, block_len, d_head)

        QK_t = torch.matmul(q, k_t)
        attn = QK_t / math.sqrt(q.size(-1))

        # Todo: Masking
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, block_len, -1)

        return attn


class GlobalAttention(nn.Module):
    """Original author: Jake Tae
    Modified: Yijing Feng (Add mask to QEr)
    """

    def __init__(self, n_embd, n_head, max_len=1024, dropout=0.1, rel_pe=True):
        super().__init__()
        d_head, remainder = divmod(n_embd, n_head)
        if remainder:
            raise ValueError("incompatible `n_embd` and `n_head`")

        self.max_len = max_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)

        self.rel_pe = rel_pe
        if self.rel_pe:
            # Er can be shared across heads
            self.Er = nn.Parameter(torch.randn(max_len, d_head))
        else:
            self.Er = None

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x, mask=None):
        # x.shape == (batch_size, seq_len, n_embd)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        k_t = self.key(x).reshape(batch_size, seq_len,
                                  self.n_head, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, n_head, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len,
                                  self.n_head, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len,
                                  self.n_head, -1).transpose(1, 2)
        # shape = (batch_size, n_head, seq_len, d_head)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, n_head, seq_len, seq_len)

        if self.rel_pe:
            attn = QK_t / math.sqrt(q.size(-1))

        else:
            start = self.max_len - seq_len
            Er_t = self.Er[start:, :].transpose(0, 1)
            # Er_t.shape = (d_head, seq_len)
            QEr = torch.matmul(q, Er_t)
            # QEr.shape = (batch_size, n_head, seq_len, seq_len)
            Srel = self._skew(QEr)
            # Srel.shape = (batch_size, n_head, seq_len, seq_len)
            attn = (QK_t + Srel) / math.sqrt(q.size(-1))

        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, n_head, seq_len, seq_len)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, n_head, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, n_head, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, n_embd)
        return self.dropout(out)

    def _mask_QEr(sel, QEr):
        _, _, seq_len, _ = QEr.shape
        mask = torch.tril(torch.ones(seq_len, seq_len)).flip(1)
        return QEr.masked_fill(mask == 0, 0)

    def _skew(self, QEr):
        # Mask upper left triangle
        QEr = self._mask_QEr(QEr)

        # QEr.shape = (batch_size, n_head, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, n_head, seq_len, 1 + seq_len)
        batch_size, n_head, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, n_head, num_cols, num_rows)
        # reshaped.size = (batch_size, n_head, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, n_head, seq_len, seq_len)
        return Srel


class LocalAttention(nn.module):

    def __init__(self, n_embd, n_head, block_len=128, dropout=0.1):

        d_head, remainder = divmod(n_embd, n_head)
        if remainder:
            raise ValueError(
                "incompatible `n_embd` and `n_head`"
            )

        self.block_len = block_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(2 * block_len - 1, d_head))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_len, block_len))
            .unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        # x.shape == (batch_size, seq_len, n_embd)
        batch_size, seq_len, _ = x.shape

        # If (seq_len < 2 * block_len), then we use only one block.
        if seq_len < 2 * self.block_len:
            block_len = seq_len
        else:
            block_len = self.block_len

        pad_len = torch.fmod(torch.tensor(-seq_len), torch.tensor(block_len))
        padded_x = torch.nn.functional.pad(x, [0, 0, 0, pad_len, 0, 0])
        n_block = (seq_len + pad_len) / block_len
        padded_x = padded_x.reshape(int(batch_size * n_block), block_len, -1)

        k_t = self.key(padded_x).reshape(batch_size, block_len,
                                         self.n_head, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, n_head, d_head, seq_len)
        v = self.value(padded_x).reshape(batch_size, block_len,
                                         self.n_head, -1).transpose(1, 2)
        q = self.query(padded_x).reshape(batch_size, block_len,
                                         self.n_head, -1).transpose(1, 2)

        QK_t = torch.matmul(q, k_t)

        # Todo: attend to previous block, debug!
        QEr = torch.matmul(q, self.Er.transpose(0, 1))
        Srel = self._skew(QEr)

        attn = (QK_t + Srel) / math.sqrt(q.size(-1))

        # Todo: masking

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, n_head, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, n_head, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, n_embd)
        return self.dropout(out)

    def _mask_QEr(self, QEr):
        _, _, block_len, _ = QEr.shape
        mask_0 = torch.triu(torch.ones(block_len, 2 * block_len - 1)).flip(1)
        mask_1 = torch.triu(torch.ones(block_len, 2 * block_len - 1)).flip(0)
        mask = mask_0 * mask_1
        return QEr.masked_fill(mask == 0, 0)

    def _skew(self, QEr):
        # Mask upper left triangle
        batch_size, n_head, block_len, _ = QEr.shape
        QEr = self._mask_QEr(QEr)

        # QEr.shape = (batch_size, n_head, seq_len, seq_len)
        padded = F.pad(QEr, (0, 1))
        # padded.shape = (batch_size, n_head, seq_len, 1 + seq_len)

        # Flatten, append block_len - 1 0s
        padded = torch.concat([padded.flatten(), torch.zeros(block_len - 1)])

        reshaped = padded.reshape(
            batch_size, n_head, block_len + 1, 2 * block_len - 1)

        # reshaped.size = (batch_size, n_head, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, :block_len, -block_len:]
        # Srel.shape = (batch_size, n_head, seq_len, seq_len)
        return Srel
