import os
import torch.nn as nn
from transformers import Trainer
from transformers import TrainingArguments

from midi_tokenizer import base_vocab_size
from data_loader import MIDIDataset, MIDITokenDataCollator

model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)


# from_pretrain = True
from_pretrain = False
model_type = 'perceiverAR'
# model_type = 'gpt2'
# model_type = 'transformer-xl'

if from_pretrain:

    if model_type == 'gpt2':
        from transformers import AutoModelWithLMHead
        model = AutoModelWithLMHead.from_pretrained("gpt2")

        # Update token embedding layers
        if model.config.vocab_size != base_vocab_size:
            # Update input embedding shape with MIDI token vocab size
            new_wte = nn.Embedding(base_vocab_size, model.config.n_embd)
            model.transformer.set_input_embeddings(new_wte)

            # Update output embedding shape with MIDI token vocab size
            model.lm_head = nn.Linear(
                model.config.n_embd, base_vocab_size, bias=False)

    elif model_type == 'transformer-xl':
        # Transformer-xl is equivalent to rel attn in local case
        from transformers import TransfoXLLMHeadModel
        from transformers.models.transfo_xl.modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax

        model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")

        # Update token embedding layers because we don't have a big dictionary
        model.transformer.word_emb = nn.Embedding(base_vocab_size,
                                                  model.config.d_embed)

        model.config.cutoffs = [base_vocab_size]  # No need to cutoff vocab
        model.config.div_val = 2
        model.crit = ProjectedAdaptiveLogSoftmax(base_vocab_size,
                                                 model.config.d_embed,
                                                 model.config.d_model,
                                                 model.config.cutoffs,
                                                 model.config.div_val)
    else:
        raise ValueError(f"Unknown pretrained model type: {model_type}")

    # Update config
    model.config.eos_token_id = 1
    model.config.bos_token_id = 1  # actually no bos
    model.config.vocab_size = base_vocab_size
else:
    if model_type == 'perceiverAR':
        from models import PerceiverAR, PerceiverARConfig
        config = PerceiverARConfig()  # Set model param here
        model = PerceiverAR(config)

    # Todo: Load customized model
    elif model_type == 'music-transformer':
        from models import MusicTransformer, MusicTransformerConfig
        config = MusicTransformerConfig()
        model = MusicTransformer(config)

    elif model_type == 'transformer-xl':
        pass

    else:
        raise ValueError(f"Unknown model: {model_type}")


# max_len = 600
max_len = 1024  # perceiverAR
q_len = 512  # perceiverAR query length
train_path = "../data/train_dataset.npy"
test_path = "../data/test_dataset.npy"

train_dataset = MIDIDataset(train_path, max_len=max_len)
test_dataset = MIDIDataset(test_path, max_len=max_len)
if model_type == 'perceiverAR':
    data_collator = MIDITokenDataCollator(
        q_len=q_len, max_len=max_len, pad_to_max_len=True, align_right=True)
else:
    data_collator = MIDITokenDataCollator()

model_output_dir = os.path.join(
    model_dir, f"{model_type}-MIDI-phrase-{max_len}")

training_args = TrainingArguments(
    output_dir=model_output_dir,  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=100,  # number of training epochs
    per_device_train_batch_size=32,  # perceiver ar: 32, the rest 8
    per_device_eval_batch_size=32,  # batch size for evaluation
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
