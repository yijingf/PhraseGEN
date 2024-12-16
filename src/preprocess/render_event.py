"""Render scores tokens to midi: expand repetitions in score and convert event tokens to midi and audio.

Outputs:
expanded note event: .json files in UNROLL_EVENT_DIR/[composer] directory 
mapping between score and performance: .json files in MIDI_DIR/[composer] directory 
midi: .mid files in MIDI_DIR/[composer] directory
audio (if required): .wav files in AUDIO_DIR/[composer] directory

Usage:
python3 render_event.py [--repeat_mode "no_repeat"]

"""
import os
import json
import scipy.io.wavfile

import sys
sys.path.append("..")
from utils.common import load_event
from utils.event import expand_score, event_to_pm

# Constant
from utils.constants import EVENT_DIR, MIDI_DIR, AUDIO_DIR


class dirConfig:
    audio_dir = AUDIO_DIR
    event_dir = EVENT_DIR
    midi_dir = MIDI_DIR


def main(event_file, repeat_mode="no_repeat", to_audio=False, fs=44100.0):

    composer = os.path.basename(os.path.dirname(event_file))
    prefix = os.path.basename(event_file).split(".")[0]

    midi_file = os.path.join(dirConfig.midi_dir, composer, f"{prefix}.mid")
    mapping_file = os.path.join(dirConfig.midi_dir, composer, f"{prefix}.json")

    # Render event to midi
    score, struct = load_event(event_file)
    event, idx_mapping = expand_score(score, struct, repeat_mode)

    with open(mapping_file, "w") as f:
        json.dump({"idx_mapping": idx_mapping, "onset": sect_onset}, f)

    # render to midi with normalized tempo by setting tp_to_bin=True
    pm, sect_onset = event_to_pm(event, tp_to_bin=True)
    pm.write(midi_file)

    if to_audio:
        # Render midi to audio
        audio = pm.fluidsynth(fs=float(fs))
        audio_file = os.path.join(dirConfig.audio_dir, composer, f"{prefix}.wav")
        scipy.io.wavfile.write(audio_file, int(fs), audio)

    return


if __name__ == "__main__":
    import argparse
    from glob import glob
    from tqdm import tqdm

    parser = argparse.ArgumentParser()

    parser.add_argument("--evend_dir", dest="event_dir", type=str, default=EVENT_DIR,
                        help=f"Directory to score events. Default to {EVENT_DIR}")
    parser.add_argument("--audio_dir", dest="audio_dir", type=str, default=AUDIO_DIR,
                        help=f"Directory to rendered audio. Default to {AUDIO_DIR}")
    parser.add_argument("--midi_dir", dest="midi_dir", type=str, default=MIDI_DIR,
                        help=f"Directory to output midi. Default to {MIDI_DIR}")
    parser.add_argument("--fs", dest="fs", type=float, default=44100.0,
                        help="Rendered audio sampling frequency.")
    parser.add_argument("--to_audio", dest="to_audio", action="store_true",
                        help="Render to audio. Defaults to false.")
    parser.add_argument(
        "--repeat_mode", dest="repeat_mode", type=str, default="volta_only",
        help="Determines how the repetition is expanded. 'volta_only': only repeat sections that have volta brackets; 'no_repeat': no repeat is performed, only perform the second ending brackets for sections with volta brackets; 'full': perform all the repeats. Default to 'no_repeat'.")

    args = parser.parse_args()

    repeat_modes = ["volta_only", "no_repeat", "full"]
    assert args.repeat_modes in repeat_modes, "invalid repeat mode type, see help for more details."

    dirConfig.event_dir = args.event_dir
    dirConfig.audio_dir = args.audio_dir
    dirConfig.midi_dir = args.midi_dir

    for composer in ['mozart', 'haydn', 'beethoven', 'scarlatti']:

        os.makedirs(os.path.join(dirConfig.midi_dir, composer))

        if args.to_audio:
            os.makedirs(os.path.join(dirConfig.audio_dir, "composer"))
        event_files = sorted(glob(os.path.join(EVENT_DIR, composer, "*.json")))

        for event_file in tqdm(event_files, desc=f"{composer}: "):
            main(event_file, args.repeat_mode, args.fs, args.to_audio)
