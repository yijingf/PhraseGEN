"""Render event tokens to midi and audio.

Usage: python3 render_event.py [--event path/to/event] [--output_dir output_dir]

"""
import os
import json
import scipy.io.wavfile

import sys
sys.path.append("..")
from kern_utils.common import load_event
from kern_utils.event import unroll_repeat, event_to_pm

# Constant
from kern_utils.constants import DATA_DIR


def main(event_file, volta_only=True, fs=44100.0):

    prefix = os.path.basename(event_file).split(".")[0]
    midi_file = os.path.join(DATA_DIR, "rendered_midi", f"{prefix}.mid")
    mapping_file = os.path.join(DATA_DIR, "mapping", f"{prefix}.json")
    audio_file = os.path.join(DATA_DIR, "audio", f"{prefix}.wav")

    # Render event to midi
    event, struct = load_event(event_file)
    unrolled_event, idx_mapping = unroll_repeat(event, struct, volta_only)
    pm, sect_onset = event_to_pm(unrolled_event)

    pm.write(midi_file)

    # Render midi to audio
    audio = pm.fluidsynth(fs=float(fs))
    scipy.io.wavfile.write(audio_file, int(fs), audio)

    with open(mapping_file, "w") as f:
        json.dump({"idx_mapping": idx_mapping, "onset": sect_onset}, f)

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--event", dest="event_file", type=str,
                        help="Input event file name.")
    parser.add_argument("--output_dir", dest="output_dir", type=str,
                        help="Output path.")
    parser.add_argument("--fs", dest="fs", type=float,
                        default=44100.0, help="Rendered audio sampling frequency.")
    parser.add_argument("--unroll_all", dest="unroll_all",
                        action="store_false", help="Unroll all repeats. Default to False.")

    args = parser.parse_args()

    volta_only = not args.unroll_all
    main(args.event_file, args.output_dir, volta_only=volta_only, fs=args.fs)
