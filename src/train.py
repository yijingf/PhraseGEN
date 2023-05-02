import os
import torch.nn as nn
from transformers import Trainer
from transformers import TrainingArguments, AutoModelWithLMHead

from midi_tokenizer import base_vocab_size
from data_loader import MIDIDataset, MIDITokenDataCollator

os.makedirs('../models', exist_ok=True)

max_len = 600
train_path = "../data/train_dataset.npy"
test_path = "../data/test_dataset.npy"

train_dataset = MIDIDataset(train_path, max_len=max_len)
test_dataset = MIDIDataset(test_path, max_len=max_len)
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
    output_dir="../models/gpt2-MIDI-phrase",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=10,  # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=16,  # batch size for evaluation
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
