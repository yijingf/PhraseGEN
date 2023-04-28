import os
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from transformers import Trainer
from transformers import TrainingArguments, AutoModelWithLMHead

from midi_tokenizer import pad_id, base_vocab_size

os.makedirs('../models', exist_ok=True)


class MIDIDataset(Dataset):
    """Load preprocessed data, return audio encoding, midi tokens and features at different hierarchies.
    """

    def __init__(self, token_path):

        # Load preprocessed data
        self.tokens = np.load(token_path)

    def __getitem__(self, index):
        """Return 

        Args:
            index (int): index of entry
        """
        token = self.tokens[index]
        return token

    def __len__(self):
        return len(self.tokens)


def _torch_collate_batch(examples):
    """Collate `examples` into a batch"""

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    return torch.stack(examples, dim=0)


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


train_dataset = MIDIDataset("../data/train_dataset.npy")
test_dataset = MIDIDataset("../data/test_dataset.npy")
data_collator = MIDITokenDataCollator()

model = AutoModelWithLMHead.from_pretrained("gpt2")

if model.config.vocab_size != base_vocab_size:

    # Update config
    model.config.eos_token_id = 1
    model.config.bos_token_id = 1516
    model.config.vocab_size = base_vocab_size

    # Update input embedding shape with MIDI token vocab size
    new_wte = nn.Embedding(base_vocab_size, model.config.n_embd)
    model.transformer.set_input_embeddings(new_wte)

    # Update output embedding shape with MIDI token vocab size
    new_lm_head = nn.Linear(model.config.n_embd, base_vocab_size, bias=False)
    model.lm_head = new_lm_head

training_args = TrainingArguments(
    output_dir="../models/gpt2-MIDI-phrase2",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=10,  # number of training epochs
    per_device_train_batch_size=4,  # batch size for training
    per_device_eval_batch_size=4,  # batch size for evaluation
    eval_steps=2000,  # Number of update steps between two evaluations.
    save_steps=2500,  # after # steps model is saved
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
trainer.save_model()
