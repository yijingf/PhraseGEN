import torch
import numpy as np
from torch.utils.data import Dataset

from midi_tokenizer import pad_id


def _torch_collate_batch(examples, pad_id=pad_id):
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

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], pad_id)
    for i, example in enumerate(examples):
        result[i, : example.shape[0]] = example

    return result


class MIDITokenDataCollator():
    """
    Data collator
    """

    def __init__(self):
        pass

    def __post_init__(self):
        pass

    def __call__(self, examples, pad_id=0):
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {"input_ids": _torch_collate_batch(examples)}

        labels = batch["input_ids"].clone()
        labels[labels == pad_id] = -100
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

        # Load preprocessed data
        self.tokens = np.load(token_path, allow_pickle=True)
        self.tokens = [i[: max_len] for i in self.tokens]

    def __getitem__(self, index):
        """Return 

        Args:
            index (int): index of entry
        """
        token = self.tokens[index]
        return token

    def __len__(self):
        return len(self.tokens)
