import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def _torch_mask_collate_batch(examples, pad_id=0):

    if isinstance(examples[0][0], (list, tuple, np.ndarray)):
        inputs = [torch.tensor(e[0], dtype=torch.long) for e in examples]
        labels = [torch.tensor(e[1], dtype=torch.long) for e in examples]

    length_of_first = inputs[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(
        x.size(0) == length_of_first for x in inputs)
    if are_tensors_same_length:
        return torch.stack(inputs, dim=0), torch.stack(labels, dim=0)

    max_len = max(int(x.size(0)) for x in inputs)
    input_result = inputs[0].new_full([len(examples), max_len], pad_id)
    label_result = labels[0].new_full([len(examples), max_len], pad_id)

    for i in range(len(examples)):
        seq_len = inputs[i].shape[0]
        input_result[i, :seq_len] = inputs[i]
        label_result[i, :seq_len] = labels[i]

    return input_result, label_result


def _torch_collate_batch(examples, pad_id=0,
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


class MaskedKrnDataCollator():
    """
    Data collator
    """

    def __init__(self, mask_id=4, pad_id=0, is_mass=False):

        self.mask_id = mask_id
        self.pad_id = pad_id
        self.is_mass = is_mass

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.

        input_ids, labels = _torch_mask_collate_batch(examples,
                                                      pad_id=self.pad_id)

        batch = {"input_ids": input_ids}

        if self.is_mass:
            decoder_input_ids = torch.ones_like(labels)
            decoder_input_ids[:, 0] = self.mask_id
            decoder_input_ids[:, 1:] = labels[:, :-1]
            batch['decoder_input_ids'] = decoder_input_ids

        labels[labels == self.pad_id] = -100
        labels[labels == self.mask_id] = -100
        # nn.CrossEntropy ignore pad_id by default
        batch["labels"] = labels

        return batch


class KrnDataCollator():
    """
    Data collator
    """

    def __init__(self, pad_id=0, q_len=None, max_len=256,
                 pad_to_max_len=False, align_right=False):
        self.q_len = q_len
        self.pad_id = pad_id
        self.max_len = max_len
        self.pad_to_max_len = pad_to_max_len
        self.align_right = align_right

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {"input_ids": _torch_collate_batch(examples,
                                                   pad_id=self.pad_id)}

        labels = batch["input_ids"].clone()

        if self.q_len is not None:
            labels = labels[:, -self.q_len:]
            batch["q_len"] = self.q_len

        # if self.mask_pad:
        #     key_padding_mask = torch.ones_like(labels)
        #     key_padding_mask[labels == self.pad_id] = 0
        #     batch['key_padding_mask'] = key_padding_mask

        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore pad_id by default
        batch["labels"] = labels

        return batch


class KrnDataset(Dataset):
    """Load preprocessed data, return audio encoding, midi tokens and features at different hierarchies.
    """

    def __init__(self, token_path, max_len=256, shuffle=True):

        with open(token_path) as f:
            tokens = json.load(f)

        self.tokens = []
        for phrase in tokens:
            if len(phrase) <= max_len:
                self.tokens.append(phrase)

        if shuffle:
            idx = list(range(len(self.tokens)))
            random.shuffle(idx)
            self.tokens = [self.tokens[i] for i in idx]

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


class MaskedKrnDataset(Dataset):
    def __init__(self, token_path, seq_len=512, shuffle=True):

        with open(token_path) as f:
            tokens = json.load(f)

        tokens = [i for i in tokens if len(i['input_ids']) <= seq_len]
        self.input_tokens = [i['input_ids'] for i in tokens]
        self.labels = [i['labels'] for i in tokens]

        if shuffle:
            idx = list(range(len(tokens)))
            random.shuffle(idx)
            self.input_tokens = [self.input_tokens[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]

    def __getitem__(self, index):
        """Return 

        Args:
            index (int): index of entry
        """
        input_ids = self.input_tokens[index]
        labels = self.labels[index]
        return input_ids, labels

    def __len__(self):
        return len(self.input_tokens)
