import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def _mask_collate_batch(examples, pad_id=0, mask_id=4, pred_masked_only=False):
    """
    Apply bar-level masking.
    """
    tokens, mask_idx = [], []
    for i, e in enumerate(examples):
        if len(e[1]):
            tokens.append(e[0])
            mask_idx.append(e[1])

    if isinstance(examples[0][0], (list, tuple, np.ndarray)):
        tokens = [torch.tensor(e, dtype=torch.long) for e in tokens]

    max_len = max(int(x.size(0)) for x in tokens)
    input_result = tokens[0].new_full([len(tokens), max_len], pad_id)
    label_result = tokens[0].new_full([len(tokens), max_len], pad_id)
    decoder_input = tokens[0].new_full([len(tokens), max_len], mask_id)

    for i in range(len(tokens)):
        seq_len = tokens[i].shape[0]
        input = tokens[i].clone()
        input[mask_idx[i]] = mask_id
        input_result[i, :seq_len] = input

        decoder_input[i, mask_idx[i]] = tokens[i][mask_idx[i]]

        if pred_masked_only:
            label_result[i, mask_idx[i]] = tokens[i][mask_idx[i]]
        else:
            label_result[i, :seq_len] = tokens[i]

    pad = torch.tensor([[mask_id] for _ in range(len(tokens))],
                       dtype=torch.long)
    decoder_input = torch.cat((pad, decoder_input[:, :-1]), 1)

    return input_result, decoder_input, label_result


class MassDataCollator():
    """
    Data collator
    """

    def __init__(self, mask_id=4, pad_id=0, pred_masked_only=False, mask_pad=True):

        self.mask_id = mask_id
        self.pad_id = pad_id
        self.pred_masked_only = pred_masked_only
        self.mask_pad = mask_pad

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.

        inputs, decoder_inputs, labels = _mask_collate_batch(examples,
                                                             pad_id=self.pad_id,
                                                             mask_id=self.mask_id,
                                                             pred_masked_only=self.pred_masked_only)

        batch = {"input_ids": inputs}

        batch['decoder_input_ids'] = decoder_inputs

        if self.mask_pad:
            # Encoder Attention Mask, equivalent to key_padding_mask
            attention_mask = torch.ones_like(inputs)
            attention_mask[inputs == self.pad_id] = 0
            batch['attention_mask'] = attention_mask

            # Decoder Attention Mask, equivalent to key_padding_mask
            decoder_attention_mask = torch.ones_like(decoder_inputs)
            attention_mask[decoder_inputs == self.pad_id] = 0
            batch['decoder_attention_mask'] = decoder_attention_mask

        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore pad_id by default
        batch["labels"] = labels

        return batch


def _dynamic_mask_collate_batch(examples, pad_id=0, mask_id=4,
                                vocab_size=324,
                                corrupt_ratio=.15, mask_ratio=.8, replace_ratio=.1):
    """
    Collate `examples` into a batch, adapted from huggingface transformer
    Randomly corrupt individual tokens in a sequence.
    """
    # Assume examples[0][1] is invalid Mask
    tokens = []
    for i, e in enumerate(examples):
        tokens.append(e[0])

    # Tensorize if necessary.
    if isinstance(tokens[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in tokens]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(
        x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    max_len = max(x.size(0) for x in examples)

    input_ids = examples[0].new_full([len(examples), max_len], pad_id)
    labels = examples[0].new_full([len(examples), max_len], pad_id)

    for i, example in enumerate(examples):

        seq_len = example.shape[0]

        label = example.clone()

        n_corrupt = int(seq_len * corrupt_ratio)
        corrupt_idx = random.sample(range(seq_len), n_corrupt)

        n_mask = int(n_corrupt * mask_ratio)
        mask_idx = corrupt_idx[: n_mask]

        n_replace = int(n_corrupt * replace_ratio)
        replace_idx = corrupt_idx[n_mask: n_mask + n_replace]
        replaced = random.choices(range(pad_id + 1, vocab_size), k=n_replace)

        labels[i, replace_idx] = label[replace_idx]
        labels[i, mask_idx] = label[mask_idx]

        input_ids[i, :seq_len] = example
        input_ids[i, replace_idx] = torch.tensor(replaced, dtype=torch.long)
        input_ids[i, mask_idx] = mask_id

    return input_ids, labels


class BertDataCollator():
    """
    Bert Data Corruption
    The training data generator chooses 15% of the token positions at random for prediction. 
    If the i-th token is chosen, we replace the i-th token with 
    (1) the [MASK] token 80% of the time
    (2) a random token 10% of the time
    (3) the unchanged i-th token 10% of the time. 
    Then, Ti will be used to predict the original token with cross entropy loss. 
    """

    def __init__(self, pad_id=0, mask_id=4, max_len=512, vocab_size=324,
                 corrupt_ratio=.15, mask_ratio=.8, replace_ratio=.1, mask_pad=True):

        self.pad_id = pad_id
        self.mask_id = mask_id
        self.max_len = max_len

        self.vocab_size = vocab_size
        self.corrupt_ratio = corrupt_ratio

        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio

        self.mask_pad = mask_pad

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {}
        input_ids, labels = _dynamic_mask_collate_batch(examples,
                                                        pad_id=self.pad_id,
                                                        mask_id=self.mask_id,
                                                        vocab_size=self.vocab_size,
                                                        corrupt_ratio=self.corrupt_ratio,
                                                        mask_ratio=self.mask_ratio,
                                                        replace_ratio=self.replace_ratio)

        batch['input_ids'] = input_ids

        if self.mask_pad:
            # Attention Mask, equivalent to key_padding_mask
            attention_mask = torch.ones_like(input_ids)
            attention_mask[input_ids == self.pad_id] = 0
            batch['attention_mask'] = attention_mask

        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore pad_id by default
        batch["labels"] = labels

        return batch


def _collate_batch(examples, pad_id=0, max_len=512,
                   pad_to_max_len=False, align_right=False):
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


class BaseDataCollator():
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
        batch = {"input_ids": _collate_batch(examples, pad_id=self.pad_id)}

        labels = batch["input_ids"].clone()

        if self.q_len is not None:
            labels = labels[:, -self.q_len:]
            batch["q_len"] = self.q_len

        key_padding_mask = torch.ones_like(labels)
        key_padding_mask[labels == self.pad_id] = 0
        batch['key_padding_mask'] = key_padding_mask

        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore pad_id by default
        batch["labels"] = labels

        return batch


class BaseDataset(Dataset):
    def __init__(self, token_path, seq_len=512, shuffle=True):

        with open(token_path) as f:
            phrases = json.load(f)

        self.input_tokens = [
            i['token_ids'] for i in phrases if len(i['token_ids']) <= seq_len]

        if shuffle:
            idx = list(range(len(self.input_tokens)))
            random.shuffle(idx)
            self.input_tokens = [self.input_tokens[i] for i in idx]

    def __getitem__(self, index):
        """Return

        Args:
            index (int): index of entry
        """
        token_ids = self.input_tokens[index]
        return token_ids

    def __len__(self):
        return len(self.input_tokens)


class MaskedDataset(Dataset):
    def __init__(self, token_path, seq_len=512, shuffle=True, mask_mode='center'):

        with open(token_path) as f:
            phrases = json.load(f)

        phrases = [i for i in phrases if len(i['token_ids']) <= seq_len]

        self.input_tokens = [i['token_ids'] for i in phrases]
        if mask_mode == 'center':
            self.mask_idx = [i['center_mask_idx'] for i in phrases]
        elif mask_mode == 'rand':
            self.mask_idx = [i['rand_mask_idx'] for i in phrases]
        elif mask_mode == 'mix':
            self.input_tokens += self.input_tokens
            self.mask_idx = [i['rand_mask_idx']
                             for i in phrases] + [i['center_mask_idx'] for i in phrases]
        else:
            raise ValueError("Unknown masking type.")

        if shuffle:
            idx = list(range(len(self.input_tokens)))
            random.shuffle(idx)
            self.input_tokens = [self.input_tokens[i] for i in idx]
            self.mask_idx = [self.mask_idx[i] for i in idx]

    def __getitem__(self, index):
        """Return

        Args:
            index (int): index of entry
        """
        token_ids = self.input_tokens[index]
        mask_idx = self.mask_idx[index]
        return token_ids, mask_idx

    def __len__(self):
        return len(self.input_tokens)
