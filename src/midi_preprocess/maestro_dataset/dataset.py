import os
import json
import pretty_midi
import pandas as pd

import sys
sys.path.append("../..")

# Constants
from midi_utils.constants import DATA_DIR, MAESTRO_DIR

MIDI_PHRASE_DIR = os.path.join(DATA_DIR, "midi_phrase")


def load_phrase_idx(fname="phrase_index.json"):

    phrase_index_file = os.path.join(DATA_DIR, fname)

    with open(phrase_index_file) as f:
        phrase_idx = json.load(f)

    phrase_idx_dict = {}
    for i in sorted(phrase_idx.keys()):
        phrase_idx_dict[int(i)] = phrase_idx[i]

    return phrase_idx_dict


def main(token_type='abs', add_eos=True):

    df = pd.read_csv(os.path.join(MAESTRO_DIR, "maestro-v3.0.0.csv"))
    phrase_idx_dict = load_phrase_idx()
    df = df[df.index.isin(phrase_idx_dict.keys())]

    if token_type == 'abs':
        from midi_utils.tokenizer import MT3Tokenizer
        tokenizer = MT3Tokenizer(add_eos=add_eos)
        token_dir = os.path.join(DATA_DIR, "abs_token")
    else:
        from midi_utils.rel_tokenizer import RelTokenizer
        tokenizer = RelTokenizer(add_eos=add_eos)
        token_dir = os.path.join(DATA_DIR, "rel_token")

    for split in ['train', 'split', 'validation']:

        data = []
        piece_idx = df[df == split].index
        midi_files = [
            f"{i}-{j}.mid" for i in piece_idx for j in phrase_idx_dict[i]['phrase_idx']]

        for midi_file in midi_files:
            pm = pretty_midi.pretty_midi(midi_file)
            token_ids = tokenizer.encode_pm(pm)

            if len(token_ids):
                data.append({"token_ids": token_ids})

        token_file = os.path.join(token_dir, f"{split}.json")
        with open(token_file, "w") as f:
            json.dump(data, f)


if __name__ == "__main__":

    main()
