import os
import torch.nn as nn
from transformers import Trainer
from transformers import TrainingArguments

from midi_tokenizer import base_vocab_size as abs_vocab_size
from midi_rel_tokenizer import base_vocab_size as rel_vocab_size
from data_loader import MIDIDataset, MIDIDataCollator


def build_model(model_type='perceiverAR', max_len=1024, from_pretrained=False, token_type='abs'):
    if token_type == 'rel':
        vocab_size = rel_vocab_size
    elif token_type == 'abs':
        vocab_size = abs_vocab_size
    else:
        raise ValueError(f"Invalid token type: {token_type}")

    if model_type == 'gpt2':
        if from_pretrained:
            from transformers import AutoModelWithLMHead
            model = AutoModelWithLMHead.from_pretraineded("gpt2")

            # Update token embedding layers
            if model.config.vocab_size != vocab_size:
                # Update input embedding shape with MIDI token vocab size
                new_wte = nn.Embedding(vocab_size, model.config.n_embd)
                model.transformer.set_input_embeddings(new_wte)

                # Update output embedding shape with MIDI token vocab size
                model.lm_head = nn.Linear(model.config.n_embd,
                                          vocab_size, bias=False)
        else:
            from transformers import GPT2LMHeadModel, GPT2Config
            config = GPT2Config(vocab_size=vocab_size)
            model = GPT2LMHeadModel(config)

    elif model_type == 'perceiverAR':
        if from_pretrained:
            raise ValueError(f"No pretrained model available for {model_type}")
        else:
            from models import PerceiverAR, PerceiverARConfig
            config = PerceiverARConfig(
                vocab_size=vocab_size)  # Set model param here
            model = PerceiverAR(config)

    elif model_type == 'music-transformer':
        if from_pretrained:
            raise ValueError(f"No pretrained model available for {model_type}")
        else:
            from models import MusicTransformer, MusicTransformerConfig
            config = MusicTransformerConfig(vocab_size=vocab_size,
                                            n_positions=max_len,
                                            n_head=8,
                                            n_layer=8,
                                            pdrop=.1,)
            model = MusicTransformer(config)

    # Transformer-xl is equivalent to rel attn in local case
    elif model_type == 'transformer-xl':
        from transformers import TransfoXLLMHeadModel
        if from_pretrained:
            from transformers.models.transfo_xl.modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax
            model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
            # Update token embedding layers because we don't have a big dictionary
            model.transformer.word_emb = nn.Embedding(vocab_size,
                                                      model.config.d_embed)
            model.config.cutoffs = [vocab_size]  # No need to cutoff vocab
            model.config.div_val = 2
            model.crit = ProjectedAdaptiveLogSoftmax(vocab_size,
                                                     model.config.d_embed,
                                                     model.config.d_model,
                                                     model.config.cutoffs,
                                                     model.config.div_val)
        else:
            from transformers import TransfoXLConfig
            config = TransfoXLConfig(vocab_size=vocab_size,
                                     cutoffs=[vocab_size],
                                     div_val=2,
                                     n_layer=12)
            model = TransfoXLLMHeadModel(config)

    else:
        raise ValueError(f"Unknown pretrained model type: {model_type}")

    # Update config
    if from_pretrained:
        model.config.eos_token_id = 1
        model.config.bos_token_id = 1  # actually no bos
        model.config.vocab_size = vocab_size

    return model


def main(train_path, eval_path, max_len=1024, q_len=512, mask_pad=True,
         batch_size=32, n_epochs=100, model_type='perceiverAR',
         reverse_token=False, token_type='abs',
         from_pretrained=False, model_dir="../models", checkpoint_path=None):

    # Build model
    model = build_model(model_type=model_type,
                        max_len=max_len,
                        token_type=token_type,
                        from_pretrained=from_pretrained)

    # Load Dataset
    train_dataset = MIDIDataset(
        train_path, max_len=max_len, reverse=reverse_token)
    eval_dataset = MIDIDataset(
        eval_path, max_len=max_len, reverse=reverse_token)

    if model_type == 'perceiverAR':
        data_collator = MIDIDataCollator(
            q_len=q_len, max_len=max_len, pad_to_max_len=True, align_right=True, mask_pad=mask_pad)
    else:
        data_collator = MIDIDataCollator(max_len=max_len, mask_pad=mask_pad)

    # Setup Training Args
    os.makedirs(model_dir, exist_ok=True)
    model_output_dir = os.path.join(model_dir,
                                    f"{model_type}-MIDI-phrase-{max_len}")

    if reverse_token:
        model_output_dir = f"{model_output_dir}-reverse"

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
    parser.add_argument("-l", dest="seq_len", type=int,
                        default=1024, help="max input sequence length")
    parser.add_argument("-r", "--reverse", dest="reverse",
                        action="store_true", help="train on reversed tokens")
    parser.add_argument("--bs", dest="batch_size", type=int,
                        default=32, help="batch size")
    parser.add_argument("--token_type", dest="token_type", type=str,
                        default="abs", help="MIDI token type, either abs or rel")
    args = parser.parse_args()

    # max_len = 600
    # max_len = 1024  # perceiverAR, music-transformer
    # q_len = 512  # perceiverAR query length

    train_path = "../data/abs_tokens/train.json"
    eval_path = "../data/abs_tokens/eval.json"

    # train_path = "../data/rel_tokens/orig_train_dataset.npy"
    # eval_path = "../data/rel_tokens/orig_test_dataset.npy"

    main(train_path, eval_path,
         model_type=args.model_type,
         batch_size=args.batch_size,
         mask_pad=True,
         max_len=args.seq_len,
         token_type=args.token_type,
         reverse_token=args.reverse)
