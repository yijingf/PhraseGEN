import torch
import numpy as np
from torch.utils.data import Dataset

from midi_tokenizer import pad_id


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
            result[i, : example.shape[0]] = example

    return result


class MIDITokenDataCollator():
    """
    Data collator
    """

    def __init__(self, pad_id=pad_id, q_len=None, max_len=1024, pad_to_max_len=False, align_right=False):
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
                                                   pad_id=self.pad_id,
                                                   max_len=self.max_len,
                                                   pad_to_max_len=self.pad_to_max_len,
                                                   align_right=self.align_right)}

        labels = batch["input_ids"].clone()

        if self.q_len is not None:
            labels = labels[:, -self.q_len:]
            batch["q_len"] = self.q_len

        labels[labels == self.pad_id] = -100
        batch["labels"] = labels

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

    def __init__(self, token_path, max_len=600):

        # Load midi token sequences
        self.tokens = np.load(token_path, allow_pickle=True)

        # Remove sequence longer than max_len
        self.tokens = [i for i in self.tokens if len(i) <= max_len]
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
