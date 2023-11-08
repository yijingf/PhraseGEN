"""Render event tokens to midi and audio.

Usage: python3 render_event.py [--event path/to/event] [--output_dir output_dir]

"""
import os
import json
import pretty_midi
import scipy.io.wavfile
from fractions import Fraction

import sys
sys.path.append("..")
from kern_utils.decode import decode_token_to_pm
from kern_utils.common import normalize_ts, load_event
from kern_utils.event import remove_repeat, concat_event, get_sub_sect_event


def unroll_repeat(event_file, volta_only=True):
    """Unroll repeats according to the section pattern in event_file.

    Args:
        event_file (str): Path to event file.
        volta_only (bool, optional): Unroll repeats only if there is volta bracket. Defaults to True.
    """
    event, struct = load_event(event_file)

    # Sort sections
    onsets = sorted([(i, v) for i, v in struct['attr'].items()],
                    key=lambda x: (x[1]['idx'], x[1]['onset']))
    sub_sect_event = get_sub_sect_event(event, onsets)

    if volta_only:
        # Unroll repeats only if there is a volta.
        sects = remove_repeat(struct['pattern'])
    else:
        sects = struct['pattern']

    unrolled_event, idx_mapping = concat_event(sub_sect_event,
                                               sects,
                                               struct['attr'])
    return unrolled_event, idx_mapping


def event_to_pm(event):
    i_st = min(event)
    i_ed = max(event)

    sect_onset = []
    tokens = []
    prev_tp, prev_ts = None, None

    t_offset, next_t_offset = 0, 0
    inst = pretty_midi.Instrument(program=0)

    # Convert unrolled event to midi
    for i in range(i_st, i_ed + 1):

        # Normalize ts, tp
        tp = int(event[i]['tempo'] / 12) * 12
        ts = normalize_ts(event[i]['time_signature'])

        event[i]['tempo'] = tp
        event[i]['time_signature'] = ts

        if tp != prev_tp or ts != prev_ts:

            # Make a record of where tempo/time signature changes
            sect_onset.append({"measure": i, "t": next_t_offset,
                               "tempo": tp, "time_signature": ts})
            prev_tp, prev_ts = tp, ts

            if len(tokens):
                pm = decode_token_to_pm(tokens, t_offset=t_offset)
                inst.notes += pm.instruments[0].notes

            tokens = [f"ts-{ts}", f"tp-{tp}"]
            t_offset = next_t_offset

        tokens += event[i]['event'] + ['bar']
        if len(ts.split("/")) > 2:
            ts_frac = Fraction(ts[:-2]) / 4
        else:
            ts_frac = Fraction(ts)
        t_measure = int(ts_frac * 4) * 60 / tp
        next_t_offset += t_measure

    if len(tokens):
        pm = decode_token_to_pm(tokens, t_offset=t_offset)
        inst.notes += pm.instruments[0].notes

    final_pm = pretty_midi.PrettyMIDI()
    final_pm.instruments.append(inst)

    return final_pm, sect_onset


def main(event_file, output_dir, volta_only=True, fs=44100.0):

    prefix = os.path.basename(event_file).split(".")[0]
    midi_file = os.path.join(output_dir, f"{prefix}.mid")
    mapping_file = os.path.join(output_dir, f"{prefix}.json")
    audio_file = os.path.join(output_dir, f"{prefix}.wav")

    # Render event to midi
    unrolled_event, idx_mapping = unroll_repeat(event_file, volta_only)
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
