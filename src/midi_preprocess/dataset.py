"""
Make dataset for music transformer, encoder-decoder model.

Output: 
train/validation dataset as JSON file stored in DATA_DIR/rel_mt_dataset

    ```
    [
        {"token_ids": []},
        ...
    ]
    ```

Usage: 
# Generate dataset with sequence length of 512, hop size of 128.
python3 dataset.py [--seq_len 512] [--hop_size 128]

"""

import os
import json
import pandas as pd

import sys
sys.path.append("..")

# Constant
from kern_utils.constants import DATA_DIR

TOKEN_DIR = os.path.join(DATA_DIR, "rel_mt_token")


def segment(file_list, seq_len=512, hop_size=128):
    data = []

    for file in file_list:
        token_file = os.path.join(TOKEN_DIR, file)

        with open(token_file) as f:
            token_ids = json.load(f)

        total_len = len(token_ids)

        for i_st in range(0, total_len, hop_size):
            i_ed = min(i_st + seq_len, total_len)
            data.append({"token_ids": token_ids[i_st: i_ed]})

    return data


def main(seq_len=512, hop_size=128):

    fname_data_split = os.path.join(DATA_DIR, "train_val_split.csv")
    if not os.path.exists(fname_data_split):
        raise ValueError(f"{fname_data_split} not found.")
    else:
        df = pd.read_csv(fname_data_split)

    for split in ['train', 'val']:
        file_list = df[df['split'] == split]['fname'].tolist()
        data = segment(file_list, seq_len, hop_size)

        dataset_file = os.path.join(DATA_DIR,
                                    "rel_mt_dataset",
                                    f"{split}_{seq_len}.json")
        with open(dataset_file, "w") as f:
            json.dump(data, f)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", dest="seq_len", type=int,
                        default=512, help="Max sequence length.")
    parser.add_argument("--hop_size", dest="hop_size", type=int,
                        default=128, help="Hop size.")

    args = parser.parse_args()
    main(args.seq_len, args.hop_size)
