import os
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter

from humdrum.common import token2v
from krn_tokenizer import BertTokenizer


def validate_phrase(phrase):
    i = 0
    while i < len(phrase):
        if phrase[i][0] == 'o':
            break
        else:
            i += 1

    j = len(phrase)
    while j > 0:
        if phrase[j - 1][0] == 'd':
            break
        elif phrase[j - 1] == 'bar':
            break
        elif phrase[j - 1] == 'eos':
            break
        else:
            j -= 1
    return phrase[i:j]


def split_phrase(phrase, seq_len=256):
    ts_tp, tokens = phrase[:2], phrase[2:]
    bar_pos = np.where(np.array(tokens) == 'bar')[0]

    phrases = []
    st = 0
    for i in range(1, len(bar_pos)):
        if bar_pos[i] - st + 3 > seq_len:
            segment = validate_phrase(tokens[st: bar_pos[i - 1] + 1])
            if len(segment) > seq_len - 1:
                segment = validate_phrase(segment[-seq_len + 2:])

            phrases.append(ts_tp + segment)
            st = bar_pos[max(0, i - 2)]

    if len(tokens) - st + 2 > seq_len:
        segment = ts_tp + validate_phrase(tokens[-seq_len + 3:])
    else:
        segment = ts_tp + validate_phrase(tokens[st:])

    phrases.append(segment)
    return phrases


def make_dataset(tokenizer, df, data_dir, split='train', seq_len=256):
    dataset = []
    for _, row in df[df['split'] == split].iterrows():
        fname = os.path.join(data_dir, row['fname'])

        with open(fname) as f:
            phrases = json.load(f)

        for phrase in phrases['token']:
            if len(phrase) < 3:
                continue
            phrase += [tokenizer.eos_token]
            if len(phrase) <= seq_len:
                dataset.append(phrase)
            else:
                splitted_phrases = split_phrase(phrase, seq_len=seq_len)
                dataset += splitted_phrases
    return dataset


def mask_phrase(phrase, max_measure_len=64, mask_ratio=0.8,
                bar_eos_token='sep', eos_token='eos', pad_bar=True,
                rand_mask_measure=None, center_mask_measure=None):

    tokens = [phrase['time_signature'], phrase['tempo']]

    n_measure = len(phrase['note'])
    if n_measure < 2:
        return None, None, None

    # Randomly mask out measures
    if rand_mask_measure is None:
        n_masked_meaure = int(mask_ratio * n_measure)
        rand_mask_measure = random.sample(range(n_measure), n_masked_meaure)

    if isinstance(rand_mask_measure, list):
        rand_mask_measure = np.array(sorted(rand_mask_measure))

    # Mask measures in the middle
    if center_mask_measure is None:
        if n_measure < 3:
            center_mask_measure = np.array([])
        else:
            # Mask around 50% bars in the middle of a sequence
            n_mask = int(n_measure / 2)
            mask_st_i = round(n_measure / 2 / 2)
            center_mask_measure = np.arange(mask_st_i, mask_st_i + n_mask)

    if isinstance(center_mask_measure, list):
        center_mask_measure = np.array(sorted(center_mask_measure))

    if pad_bar:
        tokens += phrase['note'][0] + ['bar']
        measure_len = [0, len(phrase['note'][0]) + 1]

        for i in range(1, n_measure - 1):
            notes = phrase['note'][i]

            pad_len = max_measure_len - len(notes)
            if len(notes) < max_measure_len:
                notes += [bar_eos_token for _ in range(pad_len)]

            tokens += notes
            tokens += ['bar']
            measure_len += [len(notes) + 1]

        tokens += phrase['note'][-1]
        tokens += [eos_token]
        measure_len += [len(phrase['note'][-1]) + 1]

    else:
        tokens += [token for bar in phrase['note'] for token in bar + ['bar']]
        tokens[-1] = eos_token

        measure_len = np.array([0] + [len(measure)
                               for measure in phrase['note']])
        measure_len += np.ones(n_measure + 1, dtype=int)
        measure_len[0] = 0

    idx = np.cumsum(measure_len)
    idx += 2

    if len(center_mask_measure):
        center_st_idx = idx[center_mask_measure]
        center_ed_idx = idx[center_mask_measure + 1]
        center_mask_idx = [j for i in range(len(center_mask_measure))
                           for j in list(range(center_st_idx[i], center_ed_idx[i]))]
    else:
        center_mask_idx = []

    rand_st_idx = idx[rand_mask_measure]
    rand_ed_idx = idx[rand_mask_measure + 1]
    rand_mask_idx = [j for i in range(len(rand_mask_measure))
                     for j in list(range(rand_st_idx[i], rand_ed_idx[i]))]

    return tokens, center_mask_idx, rand_mask_idx


def make_masked_dataset(tokenizer, df, data_dir, split,
                        seq_len=512, mask_ratio=0.75, pad_bar=True):

    dataset = []

    for _, row in df[df['split'] == split].iterrows():
        fname = os.path.join(data_dir, row['fname'])

        with open(fname) as f:
            phrases = json.load(f)

        for phrase in phrases:

            tokens, center_mask_idx, rand_mask_idx = mask_phrase(phrase,
                                                                 mask_ratio=mask_ratio,
                                                                 pad_bar=pad_bar,
                                                                 bar_eos_token=tokenizer.sep_token,
                                                                 eos_token=tokenizer.eos_token)

            if tokens is None:
                continue

            if tokenizer.has_irregular_token(tokens):
                continue

            if len(tokens) >= seq_len:  # roberta pos id starts from 1
                continue

            entry = {"token_ids": tokenizer.convert_tokens_to_ids(tokens),
                     "center_mask_idx": center_mask_idx,
                     "rand_mask_idx": rand_mask_idx}
            dataset.append(entry)

    return dataset


def split_dataset(data_dir, split_ratio=0.8):
    """Split dataset by file name.

    Args:
        data_dir (_type_): _description_
        split_ratio (float, optional): _description_. Defaults to 0.8.

    Returns:
        _type_: _description_
    """
    fname_list = []

    composers = os.listdir(data_dir)
    for composer in composers:
        fname_list += glob(os.path.join(data_dir, f'{composer}/*.json'))

    fname_list = sorted(fname_list)

    n_file = len(fname_list)

    idx = list(range(n_file))
    random.shuffle(idx)

    n_train = int(n_file * split_ratio)
    train_idx = idx[: n_train]
    val_idx = idx[n_train:]

    fname_list = ['/'.join(i.split("/")[-2:]) for i in fname_list]
    df = pd.DataFrame({'fname': fname_list})
    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'

    return df


def get_vocab(seg_dir):
    """_summary_

    Args:
        seg_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    composers = os.listdir(seg_dir)

    vocab_cnt = Counter()

    for composer in composers:

        seg_files = glob(os.path.join(seg_dir, composer, "*.json"))
        for seg_file in seg_files:

            with open(seg_file) as f:
                segments = json.load(f)

            for seg in segments:
                for measure in seg['note']:
                    vocab_cnt.update(measure)

                vocab_cnt.update([seg['time_signature']])
                vocab_cnt.update([seg['tempo']])

    # Reject 1/11, 1/5 notes or notes with occurences < 10
    tokens = list(vocab_cnt.keys())
    for token in tokens:
        if token[0] in ['o', 'd']:
            div = token2v(token).denominator

            if not div % 5 or not div % 11:
                vocab_cnt.pop(token)
                continue

        if vocab_cnt[token] < 10:
            vocab_cnt.pop(token)

    vocab = sorted(list(vocab_cnt.keys()) + ['bar'])

    return vocab


def main(root_dir, split_ratio, seq_len=512, masked=False, pad_bar=False):

    seg_dir = os.path.join(root_dir, "segment")

    # Split train/validation dataset.
    fname_data_split = os.path.join(root_dir, "train_val_split.csv")
    if not os.path.exists(fname_data_split):
        df = split_dataset(seg_dir, split_ratio)
        df.to_csv(fname_data_split, index=False)
    else:
        df = pd.read_csv(fname_data_split)

    # Get base vocabulary with time signature, tempo normalized, pitch transposed
    vocab_fname = os.path.join(root_dir, "vocab", "base_vocab.txt")
    if os.path.exists(vocab_fname):
        with open(vocab_fname) as f:
            base_vocab = f.read().splitlines()

    else:
        base_vocab = get_vocab(seg_dir)
        # Save to file
        with open(vocab_fname, "w") as f:
            for i in base_vocab:
                f.write(i + "\n")

    # Train Tokenizer
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    # Make dataset
    out_dir = os.path.join(root_dir, "dataset")
    splits = ['train', 'val']
    datasets = {}
    fnames = {}

    if masked:
        for split in splits:
            datasets[split] = make_masked_dataset(tokenizer, df, seg_dir,
                                                  split=split, seq_len=seq_len, pad_bar=pad_bar)
            if pad_bar:
                fname = f"{split}_{seq_len}_masked_pad.json"
            else:
                fname = f"{split}_{seq_len}_masked.json"
            fnames[split] = os.path.join(out_dir, fname)

    else:
        for split in splits:
            datasets[split] = make_dataset(tokenizer, df, seg_dir,
                                           split=split, seq_len=seq_len)
            fnames[split] = os.path.join(out_dir, f"{split}_{seq_len}.json")

    for split in splits:
        with open(fnames[split], "w") as f:
            json.dump(datasets[split], f)

    return


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", dest="root_dir", type=str,
                        default="../../sonata-dataset/", help="Root directory.")
    parser.add_argument("--split_ratio", dest="split_ratio", type=float,
                        default=0.8, help="Train/validation split ratio. Default to 0.8.")
    parser.add_argument("--seq_len", dest="sequence_len", type=float,
                        default=512, help="Max sequence length. Default to 512.")
    parser.add_argument("--masked", dest="masked",
                        action="store_false", help="Whether to apply masking. Default to False.")

    args = parser.parse_args()
    main(args.root_dir, args.split_ratio, args.seq_len, args.masked)
