import os
import json
import pretty_midi
import numpy as np
import pandas as pd

# from sklearn.model_selection import train_test_split

from midi_tokenizer import encode_pm
from midi_tokenizer import pad_id, eos_id, num_special_tokens
from utils.midi_utils import trim_midi, strip_midi, pretty_midi_sort


def MIDI_tokenization(midi_dir, df, t_bound_dict, token_path, idx_path=None,
                      min_t_phrase=4, max_t_phrase=32, min_n_note=8,
                      tempo_shift=1):
    """_summary_

    Args:
        midi_dir (_type_): _description_
        df (_type_): _description_
        t_bounds (_type_): _description_
        token_path (_type_): _description_
        min_t_phrase (int, optional): Minimum length of a phrase. Defaults to 4, i.e. one bar of 60BPM.
        max_t_phrase (int, optional): _description_. Defaults to 60.
        min_n_note (int, optional): Minimum notes in a phrase. Defaults to 4, i.e. 8 8th note in a bar.

    Returns:
        _type_: _description_
    """

    tokens_dict = {}
    pairs_idx_dict = {}
    phrase_idx_dict = {}

    for i, row in df.iterrows():

        fname = os.path.join(midi_dir, row['midi_filename'])

        # Load single track MIDI
        pm = pretty_midi.PrettyMIDI(fname)
        pretty_midi_sort(pm)

        # Load timestamp of phrase boundaries
        t_bound = t_bound_dict[str(i)]
        t_bound = (np.array(t_bound) / tempo_shift).tolist()

        t_interval = np.vstack([t_bound[:-1], t_bound[1:]]).T

        pair = []
        phrase_idx = []
        pairs_idx = []

        tokens = []
        for j, (t_st, t_ed) in enumerate(t_interval):

            # Skip extremely long/short segments
            t_seg = t_ed - t_st

            # Skip long segments
            if t_seg > max_t_phrase or t_seg < min_t_phrase:
                continue

            # Trim midi
            pm_seg, _ = trim_midi(pm, t_st, t_ed, cut_by='onset', meta=False)

            # Skip empty segments
            if not len(pm_seg.instruments[0].notes):
                continue

            pm_seg, strip_t = strip_midi(pm_seg)

            if pm_seg.get_end_time() < min_t_phrase:
                continue

            if strip_t < min_t_phrase:
                # Whether the current phrase is a continuation of last phrase
                if len(pair):
                    if j - pair[-1] == 1:
                        pairs_idx.append(pair + [j])
            pair = [j]

            # If this phrase is followed by `min_t_phrase` silence, then the next phrase is not likely to be it's continuation.
            if t_seg - strip_t - pm_seg.get_end_time() > min_t_phrase:
                pair = []

            # if len(pm_seg.instruments[0].notes) < min_n_note:
                # continue

            phrase_idx.append(j)

            # Tokenize MIDI segments
            events_id = encode_pm(pm_seg)

            # token = event_id + num_special_token, with eos_id appended in the end
            # token = np.append(np.array(events_id) + num_special_tokens, eos_id)
            token = np.array(events_id) + num_special_tokens

            tokens.append(token.tolist())

        tokens_dict[i] = tokens
        pairs_idx_dict[i] = pairs_idx
        phrase_idx_dict[i] = phrase_idx

    with open(token_path, "w") as f:
        json.dump(tokens_dict, f)

    idx_dict = {}
    idx_dict['pairs_idx'] = pairs_idx_dict
    idx_dict['phrase_idx'] = phrase_idx_dict

    with open(idx_path, "w") as f:
        json.dump(idx_dict, f)

    # tokens = np.array(tokens, dtype=object)
    # np.save(token_path, tokens, allow_pickle=True)
    return tokens_dict, idx_dict


def pad(tokens, max_len=600):
    n_tokens = np.array([len(i) for i in tokens])

    # Remove long sequence
    selected_tokens = tokens[np.where(n_tokens <= max_len)]
    padded_tokens = [np.concatenate([i, [pad_id for _ in range(max_len - len(i))]])
                     for i in selected_tokens]
    padded_tokens = np.array(padded_tokens)
    return padded_tokens


def main(midi_dir, df, t_bound_dict, token_path, idx_path=None, add_eos=True, len_limit=2048, tempo_shift=1):

    if not os.path.exists(token_path):
        if not os.path.exists(idx_path):
            tokens_dict, _ = MIDI_tokenization(midi_dir, df, t_bound_dict, token_path,
                                               tempo_shift=tempo_shift)
        else:
            pass
    else:
        with open(token_path) as f:
            tokens_dict = json.load(f)
        # tokens = np.load(token_path, allow_pickle=True)

    # Add EOS token
    if add_eos:
        tokens = [np.append(token, eos_id) for token in tokens]

    # Padding
    # tokens = pad(tokens, max_len)

    tokens = np.array(tokens, dtype=object)
    len_token = np.array([len(token) for token in tokens])

    # Split train test set
    # train, val = train_test_split(selected_tokens, test_size=0.2)
    return tokens[np.where(len_token <= len_limit)]


if __name__ == "__main__":

    midi_dir = "/isi/music/yijing/maestro-v3.0.0/midi/"
    df = pd.read_csv("/isi/music/yijing/maestro-v3.0.0/maestro-v3.0.0.csv")

    with open("../data/merged_phrase_boundaries.json", "r") as f:
        t_bound_dict = json.load(f)

    idx = [int(i) for i in t_bound_dict.keys()]
    df = df[df.index.isin(idx)]

    token_path = "../data/MIDI_tokens.npy"

    train, test = main(midi_dir, df, t_bound_dict, token_path)
    np.save("../data/train_dataset.npy", train)
    np.save("../data/eval_dataset.npy", test)
