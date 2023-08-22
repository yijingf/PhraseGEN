import os
import json
import random
import numpy as np
import pandas as pd

from glob import glob
from fractions import Fraction


def get_time(event):
    return Fraction(event.split('-')[-1])


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


def make_dataset(df, data_dir, split='train', seq_len=256):
    dataset = []
    for _, row in df[df['split'] == split].iterrows():
        fname = os.path.join(data_dir, row['fname'])

        with open(fname) as f:
            phrases = json.load(f)

        for phrase in phrases['token']:
            if len(phrase) < 3:
                continue
            phrase += ['eos']
            if len(phrase) <= seq_len:
                dataset.append(phrase)
            else:
                splitted_phrases = split_phrase(phrase, seq_len=seq_len)
                dataset += splitted_phrases
    return dataset


def split_dataset(fname):

    fname_list = []
    for composer in ['mozart', 'haydn', 'beethoven', 'scarlatti']:
        fname_list += glob(f'../../sonata-dataset/token/{composer}/*.json')

    fname_list = sorted(fname_list)

    n_file = len(fname_list)

    idx = list(range(n_file))
    random.shuffle(idx)

    n_train = int(n_file * 0.8)
    train_idx = idx[: n_train]
    val_idx = idx[n_train:]

    fname_list = ['/'.join(i.split("/")[-2:]) for i in fname_list]
    df = pd.DataFrame({'fname': fname_list})
    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'

    df.to_csv(fname, index=False)
    return df


def clean_token(data):
    clean_data = []

    for phrase in data:

        flag = False
        for token in phrase:
            if token[0] in ['o', 'd']:
                div = get_time(token).denominator
                if div and not div % 11:
                    flag = True
                    break

                if div and not div % 5:
                    flag = True
                    break

        if not flag:
            clean_data.append(phrase)
    return clean_data


def main(root_dir):
    return


if __name__ == "__main__":

    data_dir = "../../sonata-dataset/token"
    fname_data_split = "../../sonata-dataset/phrase_train_val_split.json"
    if not os.path.exists(fname_data_split):
        df = split_dataset(fname_data_split)
    else:
        df = pd.read_csv(fname_data_split)

    train_data = make_dataset(df, data_dir, split='train')
    val_data = make_dataset(df, data_dir, split='val')

    with open("../../sonata-dataset/train.json", "w") as f:
        json.dump(train_data, f)
