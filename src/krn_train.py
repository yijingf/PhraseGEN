import os
import torch.nn as nn
from transformers import Trainer
from transformers import TrainingArguments

from krn_tokenizer import BaseTokenizer, BertTokenizer
from krn_data_loader import KrnDataCollator, KrnDataset
from krn_data_loader import MaskedKrnDataCollator, MaskedKrnDataset


def build_model(vocab_size, model_type='mass', max_len=1024, pad_id=0):

    if model_type == 'music-transformer':
        from models import MusicTransformer, MusicTransformerConfig
        config = MusicTransformerConfig(vocab_size=vocab_size,
                                        n_positions=max_len,
                                        n_head=4,
                                        n_layer=4,
                                        pdrop=.1,)
        model = MusicTransformer(config)

    elif model_type == 'bert':
        from transformers import BertConfig, BertForMaskedLM
        config = BertConfig(vocab_size=vocab_size,
                            hidden_size=768,
                            num_hidden_layers=6,
                            num_attention_heads=8,
                            intermediate_size=1024,
                            max_position_embeddings=max_len)
        model = BertForMaskedLM(config)

    elif model_type == 'roberta':
        from transformers import RobertaConfig, RobertaForMaskedLM
        config = RobertaConfig(vocab_size=vocab_size,
                               hidden_size=768,
                               num_hidden_layers=4,
                               num_attention_heads=4,
                               intermediate_size=1024,
                               max_position_embeddings=max_len)

        model = RobertaForMaskedLM(config)

    elif model_type == 'mass':
        from transformers import EncoderDecoderModel
        from transformers import BertConfig, EncoderDecoderConfig

        config_encoder = BertConfig(vocab_size=vocab_size,
                                    num_hidden_layers=4,
                                    num_attention_heads=8,
                                    intermediate_size=1024)

        config_decoder = BertConfig(vocab_size=vocab_size,
                                    num_hidden_layers=4,
                                    num_attention_heads=8,
                                    intermediate_size=1024)

        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder,
                                                                   config_decoder)
        config.pad_token_id = pad_id
        model = EncoderDecoderModel(config)

    else:
        raise ValueError(f"Unknown pretrained model type: {model_type}")

    return model


def main(train_path, eval_path, base_vocab_file, max_len=256, bar_pad=False,
         batch_size=32, n_epochs=100, model_type='mass',
         model_dir="../models", checkpoint_path=None):

    with open(base_vocab_file) as f:
        base_vocab = f.read().splitlines()

    # Load Dataset
    if model_type not in ['bert', 'roberta', 'mass']:
        train_dataset = KrnDataset(train_path, max_len=max_len)
        eval_dataset = KrnDataset(eval_path, max_len=max_len)
        data_collator = KrnDataCollator(max_len=max_len)
        tokenizer = BaseTokenizer()
        tokenizer.train(base_vocab)
    else:
        tokenizer = BertTokenizer()
        tokenizer.train(base_vocab)
        if model_type == 'bert':
            train_dataset = MaskedKrnDataset(train_path, seq_len=max_len)
            eval_dataset = MaskedKrnDataset(eval_path, seq_len=max_len)
            data_collator = MaskedKrnDataCollator(mask_id=tokenizer.mask_id,
                                                  pad_id=tokenizer.pad_id,)

        elif model_type == 'roberta':
            train_dataset = MaskedKrnDataset(train_path, seq_len=max_len - 1)
            eval_dataset = MaskedKrnDataset(eval_path, seq_len=max_len - 1)
            data_collator = MaskedKrnDataCollator(mask_id=tokenizer.mask_id,
                                                  pad_id=tokenizer.pad_id,)

        elif model_type == 'mass':
            train_dataset = MaskedKrnDataset(train_path, seq_len=max_len)
            eval_dataset = MaskedKrnDataset(eval_path, seq_len=max_len)
            data_collator = MaskedKrnDataCollator(mask_id=tokenizer.mask_id,
                                                  pad_id=tokenizer.pad_id,
                                                  is_mass=True)

        else:
            raise ValueError('invalid model type')

    # Build model
    model = build_model(tokenizer.vocab_size,
                        model_type=model_type,
                        max_len=max_len,
                        pad_id=tokenizer.pad_id)

    # Setup Training Args
    os.makedirs(model_dir, exist_ok=True)
    if bar_pad:
        prefix = f"{model_type}-pad"
    else:
        prefix = f"{model_type}"
    model_output_dir = os.path.join(model_dir, f"{prefix}-krn-{max_len}")

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
    import json
    parser = argparse.ArgumentParser()

    parser.add_argument("--arg_file", dest="arg_file", type=str,
                        default="", help="Arguments .json file.")
    input_args = parser.parse_args()

    if input_args.arg_file:
        with open(input_args.arg_file) as f:
            args = json.load(f)

    main(**args)
