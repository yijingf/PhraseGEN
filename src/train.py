import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from transformers import Trainer
from transformers import TrainingArguments, AutoModelWithLMHead

pad_id = 0
base_vocab_size = 1517


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

# Todo: Model weights shape and config mismatch
model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")

# Update input embedding shape with MIDI token vocab size
old_wte_wgts = model.transformer.get_input_embeddings().weight.clone().detach()
new_wte_wgts = old_wte_wgts.new_zeros(base_vocab_size, old_wte_wgts.size(1))

new_wte = nn.Embedding(base_vocab_size, old_wte_wgts.size(1))
new_wte.weight.data = new_wte_wgts

model.transformer.set_input_embeddings(new_wte)

# Update output embedding shape with MIDI token vocab size
old_lm_head_wgts = model.lm_head.weight.clone().detach()
new_lm_head_wgts = old_lm_head_wgts.new_zeros(old_lm_head_wgts.size(1), base_vocab_size)

new_lm_head = nn.Embedding(old_lm_head_wgts.size(1), base_vocab_size)
new_lm_head.weight.data = new_lm_head_wgts

model.set_output_embeddings(new_lm_head)

# Update config
model.config.eos_token_id = 1
model.config.bos_token_id = 1516
model.config.vocab_size = base_vocab_size

import os
os.makedirs('../models', exist_ok=True)

training_args = TrainingArguments(
    output_dir="../models/gpt2-MIDI-phrase",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=10,  # number of training epochs
    per_device_train_batch_size=4,  # batch size for training
    per_device_eval_batch_size=4,  # batch size for evaluation
    eval_steps=400,  # Number of update steps between two evaluations.
    save_steps=800,  # after # steps model is saved
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
