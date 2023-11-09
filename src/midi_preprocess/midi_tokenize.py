"""Render events to midi, and tokenize midi transposed key to C major/minor using Magenta's Relative Tokenizer.

Output: token_ids in DATA_DIR/rel_mt_token/[composer]/*.json

Usage: python3 midi_tokenize.py

"""
import os
import json
import pretty_midi
from glob import glob

import sys
sys.path.append("..")
from midi_utils.common import change_pitch
from midi_utils.rel_tokenizer import RelTokenizer
from kern_utils.common import load_event
from kern_utils.event import unroll_repeat, event_to_pm

# Constants
from kern_utils.constants import PITCH_OFFSET_DICT, DATA_DIR
EVENT_DIR = os.path.join(DATA_DIR, "event")
MIDI_DIR = os.path.join(DATA_DIR, "rendered_midi")
MIDI_C_DIR = os.path.join(DATA_DIR, "rendered_midi_in_C")
TOKEN_DIR = os.path.join(DATA_DIR, "rel_mt_token")


def main(save_midi=True):

    tokenizer = RelTokenizer(num_velocity_bins=1)
    composers = os.listdir(os.path.join(DATA_DIR, "event"))

    for composer in composers:

        os.makedirs(os.path.join(MIDI_C_DIR, composer), exist_ok=True)

        event_files = glob(os.path.join(EVENT_DIR, composer, "*.json"))
        for event_file in event_files:

            prefix = os.path.basename(event_file).split(".")[0]

            # Load event
            event, struct = load_event(event_file)

            # Render event to midi
            midi_file = os.path.join(MIDI_DIR, composer, f"{prefix}.mid")
            if not os.path.exists(midi_file):
                unrolled_event, _ = unroll_repeat(event, struct)
                pm, _ = event_to_pm(unrolled_event)
            else:
                pm = pretty_midi.PrettyMIDI(midi_file)

            # Original Key
            key = event[event.keys()[0]]['key'].split()[0]
            key_shift = PITCH_OFFSET_DICT[key]

            # Transpose key to C
            change_pitch(pm, pitch_shift=int(key_shift), inplace=True)
            if save_midi:
                output_file = os.path.join(MIDI_C_DIR,
                                           composer,
                                           f"{prefix}.mid")
                pm.write(output_file)

            # Encode to token_ids
            token_ids = tokenizer.encode_pm(pm)
            token_file = os.path.join(TOKEN_DIR, composer, f"{prefix}.json")
            with open(token_file, "w") as f:
                json.dump(token_ids, f)

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--unroll_all", dest="unroll_all",
                        action="store_false", help="Unroll all repeats. Default to False.")

    args = parser.parse_args()

    volta_only = not args.unroll_all
    main(volta_only)
