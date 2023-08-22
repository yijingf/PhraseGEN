import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset

from midi_tokenizer import pad_id, cls_id, sep_id, mask_id, eos_id


def sliding_window(seq, seq_len=512, hop_len=256):
    n = len(seq)
    i = 0

    array = []
    while i + seq_len < n:
        array.append(seq[i: i + seq_len])
        i += hop_len

    if i < n - 1:
        array.append(seq[i:])
    return array


def make_mask(l, max_len):
    return [1 for _ in range(l)] + [0 for _ in range(max_len - l)]


def concat_phrases(tokens):
    concat_tokens = []
    token_type = []

    for phrase_0, phrase_1 in tokens:
        len_phrase_0 = len(phrase_0)
        len_phrase_1 = len(phrase_1)
        line = [cls_id] + phrase_0 + [sep_id] + phrase_1 + [sep_id]
        type_id = [0 for _ in range(len_phrase_0 + 2)] + \
            [1 for _ in range(len_phrase_1 + 1)]
        concat_tokens.append(line)
        token_type.append(type_id)

    return concat_tokens, token_type


def _torch_collate_batch(examples, pad_id=pad_id,
                         max_len=1024, pad_to_max_len=False, align_right=False):
    """Collate `examples` into a batch, adapted from huggingface transformer"""

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(
        x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    if not pad_to_max_len:
        # Creating the full tensor and filling it with our data.
        max_len = max(x.size(0) for x in examples)

    result = examples[0].new_full([len(examples), max_len], pad_id)

    for i, example in enumerate(examples):

        if align_right:
            result[i, -example.shape[0]:] = example
        else:
            result[i, :example.shape[0]] = example

    return result


class MIDIDataCollator():
    """
    Data collator
    """

    def __init__(self, pad_id=pad_id, q_len=None, max_len=1024,
                 pad_to_max_len=False, align_right=False, mask_pad=False):
        self.q_len = q_len
        self.pad_id = pad_id
        self.max_len = max_len
        self.pad_to_max_len = pad_to_max_len
        self.align_right = align_right
        self.mask_pad = mask_pad

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {"input_ids": _torch_collate_batch(examples,
                                                   pad_id=self.pad_id,
                                                   max_len=self.max_len,
                                                   pad_to_max_len=self.pad_to_max_len,
                                                   align_right=self.align_right)}

        labels = batch["input_ids"].clone()

        if self.q_len is not None:
            labels = labels[:, -self.q_len:]
            batch["q_len"] = self.q_len

        if self.mask_pad:
            key_padding_mask = torch.ones_like(labels)
            key_padding_mask[labels == self.pad_id] = self.pad_id
            batch['key_padding_mask'] = key_padding_mask

        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore pad_id by default
        batch["labels"] = labels

        return batch


class MIDINSPDataCollator():
    """
    Data collator
    """

    def __init__(self, pad_id=pad_id):
        self.pad_id = pad_id

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        n = len(examples)
        inputs = [examples[i][0] for i in range(n)]
        token_type = [examples[i][2] for i in range(n)]

        batch = {"input_ids": _torch_collate_batch(inputs,
                                                   pad_id=self.pad_id)}
        _, seq_len = batch['input_ids'].shape

        batch["labels"] = torch.tensor([examples[i][1] for i in range(n)])
        batch["token_type_ids"] = _torch_collate_batch(
            token_type, pad_id=self.pad_id)
        batch['attention_mask'] = torch.tensor(
            [make_mask(len(inputs[i]), seq_len) for i in range(n)])
        return batch


class BaseDataset(Dataset):
    def __init__(self, data):
        self.data = data
        return

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MIDIDataset(Dataset):
    """Load preprocessed data, return audio encoding, midi tokens and features at different hierarchies.
    """

    def __init__(self, token_path, max_len=512, reverse=False, shuffle=True):

        # Load midi token sequences
        # self.tokens = np.load(token_path, allow_pickle=True)

        with open(token_path) as f:
            tokens = json.load(f)

        self.tokens = []
        for _, phrases in tokens.items():
            for i in phrases:
                self.tokens += sliding_window(i + [eos_id], max_len)

        if shuffle:
            idx = list(range(len(self.tokens)))
            random.shuffle(idx)
            self.tokens = [self.tokens[i] for i in idx]

        if reverse:
            self.tokens = [i[::-1] for i in self.tokens]

        # Remove sequence longer than max_len
        # self.tokens = [i for i in self.tokens if len(i) <= max_len]
        # self.tokens = [i[: max_len] for i in self.tokens]

    def __getitem__(self, index):
        """Return 

        Args:
            index (int): index of entry
        """
        token = self.tokens[index]
        return token

    def __len__(self):
        return len(self.tokens)


class MIDINSPDataset(Dataset):
    """Load preprocessed data, return audio encoding, midi tokens and features at different hierarchies.
    """

    def __init__(self, token_path, shuffle=True):

        # Load midi token sequences
        with open(token_path) as f:
            tokens = json.load(f)

        n_pos = len(tokens['is_next'])
        n_neg = len(tokens['not_next'])
        tokens = tokens['is_next'] + tokens['not_next']

        self.tokens, self.token_type = concat_phrases(tokens)
        self.labels = [1 for _ in range(n_pos)] + [0 for _ in range(n_neg)]

        if shuffle:
            idx = list(range(len(self.tokens)))
            random.shuffle(idx)
            self.tokens = [self.tokens[i] for i in idx]
            self.token_type = [self.token_type[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]

    def __getitem__(self, index):
        """Return 

        Args:
            index (int): index of entry
        """
        token = self.tokens[index]
        label = self.labels[index]
        token_type = self.token_type[index]
        return token, label, token_type

    def __len__(self):
        return len(self.tokens)
