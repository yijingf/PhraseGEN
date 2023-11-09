"""Align annotation in seconds with beat bar position notation.

Usage: python3 align_annotation.py [--label path/to/manual/annotation] [--mapping path/to/section/index/mapping] [--output path/to/output/file]

"""

import math
import json
import pandas as pd
from fractions import Fraction


def match_annotation(label_file, sect_onset, idx_mapping):
    df = pd.read_csv(label_file, header=None, sep="\t")
    df['measure'] = None
    df['pos'] = None

    i_onset = 0
    for i_row, row in df.iterrows():
        t = row[0]

        while i_onset < len(sect_onset) - 1 and sect_onset[i_onset + 1]['t'] < t:
            i_onset += 1

        measure_offset = sect_onset[i_onset]['measure']
        t_offset = sect_onset[i_onset]['t']

        ts = sect_onset[i_onset]['time_signature']
        ts_num = int(Fraction(ts) * 4)
        tp = sect_onset[i_onset]['tempo']
        t_measure = ts_num * 60 / tp

        i_measure = int((t - t_offset) / t_measure) + measure_offset
        df.loc[i_row, 'measure'] = int(i_measure)

        pos_i = math.ceil((t - t_offset) % t_measure / t_measure * ts_num * 2)
        df.loc[i_row, 'pos'] = f"{Fraction(pos_i, 2)}"

    df['score measure'] = [idx_mapping[i] for i in df['unrolled measure']]

    return df


def main(label_file, mapping_file, output_file):

    with open(mapping_file) as f:
        meta_data = json.load(f)

    df = match_annotation(
        label_file, meta_data["onset"], meta_data["idx_mapping"])
    df.to_csv(output_file, index=False)

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--label", dest="label_file", type=str,
                        help="Input audacity label file name.")
    parser.add_argument("--mapping", dest="mapping_file", type=str,
                        help="Path to mapping file.")
    parser.add_argument("--output", dest="output_file", type=str,
                        help="Output file name.")

    args = parser.parse_args()
    main(args.label_file, args.mapping_file, args.output_file)
