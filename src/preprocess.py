import os
import json
import pretty_midi
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from midi_tokenizer import encode_pm
from midi_tokenizer import pad_id, eos_id, num_special_tokens
from utils.midi_utils import trim_midi, pretty_midi_sort


def MIDI_tokenization(midi_dir, df, t_bounds, token_path,
                      min_t_phrase=1, max_t_phrase=60):

    tokens = []

    for i, row in df.iterrows():

        fname = os.path.join(midi_dir, row['midi_filename'])

        # Load single track MIDI
        pm = pretty_midi.PrettyMIDI(fname)
        pretty_midi_sort(pm)

        # Load timestamp of phrase boundaries
        t_bound = t_bounds[str(i)]

        t_duration = pm.get_end_time()
        if t_bound[-1] >= t_duration:
            t_bound = t_bound[:-1]

        for t_st, t_ed in zip(*[[0] + t_bound, t_bound + [t_duration]]):

            # Skip extremely long/short segments
            t_duration = t_ed - t_st

            # Skip long segments
            if t_duration > max_t_phrase or t_duration < min_t_phrase:
                continue

            # Trim midi
            pm_seg, _ = trim_midi(pm, t_st, t_ed, cut_by='onset', meta=False)

            # Skip empty segments
            if not len(pm_seg.instruments[0].notes):
                continue

            # if len(pm_seg.instruments[0].notes) < min_n_note:
            #     continue

            # Tokenize MIDI segments
            events_id = encode_pm(pm_seg)

            # token = event_id + num_special_token, with eos_id appended in the end
            # token = np.append(np.array(events_id) + num_special_tokens, eos_id)
            token = np.array(events_id) + num_special_tokens

            tokens.append(token)

    tokens = np.array(tokens, dtype=object)
    np.save(token_path, tokens, allow_pickle=True)
    return tokens


def pad(tokens, max_len=600):
    n_tokens = np.array([len(i) for i in tokens])

    # Remove long sequence
    selected_tokens = tokens[np.where(n_tokens <= max_len)]
    padded_tokens = [np.concatenate([i, [pad_id for _ in range(max_len - len(i))]])
                     for i in selected_tokens]
    padded_tokens = np.array(padded_tokens)
    return padded_tokens


def main(midi_dir, df, t_bounds, token_path, add_eos=True, len_limit=1024, test_size=0.2):

    if not os.path.exists(token_path):
        tokens = MIDI_tokenization(midi_dir, df, t_bounds, token_path)
    else:
        tokens = np.load(token_path, allow_pickle=True)

    # Add EOS token
    if add_eos:
        tokens = [np.append(token, eos_id) for token in tokens]

    # Padding
    # tokens = pad(tokens, max_len)

    tokens = np.array(tokens, dtype=object)
    len_token = np.array([len(token) for token in tokens])
    selected_tokens = tokens[np.where(len_token <= len_limit)]

    # Split train test set
    train, test = train_test_split(selected_tokens, test_size=test_size)
    return train, test


if __name__ == "__main__":

    midi_dir = "/isi/music/yijing/maestro-v3.0.0/midi/"
    df = pd.read_csv("/isi/music/yijing/maestro-v3.0.0/maestro-v3.0.0.csv")

    with open("../data/phrase_boundaries.json", "r") as f:
        t_bounds = json.load(f)

    idx = [int(i) for i in t_bounds.keys()]
    df = df[df.index.isin(idx)]

    token_path = "../data/MIDI_tokens.npy"

    train, test = main(midi_dir, df, t_bounds, token_path)
    np.save("../data/train_dataset.npy", train)
    np.save("../data/test_dataset.npy", test)
