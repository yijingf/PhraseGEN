import os
import torch.nn as nn
from transformers import Trainer
from transformers import TrainingArguments

from midi_tokenizer import base_vocab_size
from data_loader import MIDIDataset, MIDITokenDataCollator


def build_model(model_type='perceiverAR', from_pretrained=False):
    if model_type == 'gpt2':
        if from_pretrained:
            from transformers import AutoModelWithLMHead
            model = AutoModelWithLMHead.from_pretraineded("gpt2")

            # Update token embedding layers
            if model.config.vocab_size != base_vocab_size:
                # Update input embedding shape with MIDI token vocab size
                new_wte = nn.Embedding(base_vocab_size, model.config.n_embd)
                model.transformer.set_input_embeddings(new_wte)

                # Update output embedding shape with MIDI token vocab size
                model.lm_head = nn.Linear(model.config.n_embd,
                                          base_vocab_size, bias=False)
        else:
            from transformers import GPT2LMHeadModel, GPT2Config
            config = GPT2Config(vocab_size=base_vocab_size)
            model = GPT2LMHeadModel(config)

    elif model_type == 'perceiverAR':
        if from_pretrained:
            raise ValueError(f"No pretrained model available for {model_type}")
        else:
            from models import PerceiverAR, PerceiverARConfig
            config = PerceiverARConfig()  # Set model param here
            model = PerceiverAR(config)

    elif model_type == 'music-transformer':
        if from_pretrained:
            raise ValueError(f"No pretrained model available for {model_type}")
        else:
            from models import MusicTransformer, MusicTransformerConfig
            config = MusicTransformerConfig()
            model = MusicTransformer(config)

    # Transformer-xl is equivalent to rel attn in local case
    elif model_type == 'transformer-xl':
        from transformers import TransfoXLLMHeadModel
        if from_pretrained:
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
            from transformers import TransfoXLConfig
            config = TransfoXLConfig(vocab_size=base_vocab_size,
                                     cutoffs=[base_vocab_size],
                                     div_val=2,
                                     n_layer=12)
            model = TransfoXLLMHeadModel(config)

    else:
        raise ValueError(f"Unknown pretrained model type: {model_type}")

    # Update config
    if from_pretrained:
        model.config.eos_token_id = 1
        model.config.bos_token_id = 1  # actually no bos
        model.config.vocab_size = base_vocab_size

    return model


def main(train_path, eval_path, max_len=1024, q_len=512,
         batch_size=32, n_epochs=100, model_type='perceiverAR',
         from_pretrained=False, model_dir="../models", checkpoint_path=None):

    # Build model
    model = build_model(model_type=model_type,
                        from_pretrained=from_pretrained)

    # Load Dataset
    train_dataset = MIDIDataset(train_path, max_len=max_len)
    eval_dataset = MIDIDataset(eval_path, max_len=max_len)

    if model_type == 'perceiverAR':
        data_collator = MIDITokenDataCollator(
            q_len=q_len, max_len=max_len, pad_to_max_len=True, align_right=True)
    else:
        data_collator = MIDITokenDataCollator()

    # Setup Training Args
    os.makedirs(model_dir, exist_ok=True)
    model_output_dir = os.path.join(model_dir,
                                    f"{model_type}-MIDI-phrase-{max_len}")

    training_args = TrainingArguments(
        output_dir=model_output_dir,  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=n_epochs,  # number of training epochs
        per_device_train_batch_size=batch_size,  # perceiver ar: 32, the rest 8
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        # eval_steps=2500,  # Number of update steps between two evaluations.
        # save_steps=2500,  # after # steps model is saved
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.save_model()
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", dest="model_type", type=str,
                        default='perceiverAR', help="model type")
    args = parser.parse_args()

    # max_len = 600
    # max_len = 1024  # perceiverAR, transformer-xl
    # q_len = 512  # perceiverAR query length
    train_path = "../data/train_dataset.npy"
    eval_path = "../data/eval_dataset.npy"

    main(train_path, eval_path, model_type=args.model_type)
