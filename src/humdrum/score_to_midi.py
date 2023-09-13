import json
import pretty_midi
from fractions import Fraction

from decode import decode_token_to_pm
from common import normalize_ts, remove_repeat
from common import load_event, get_sub_sect_event, concat_event


def unroll_repeat(event_file, volta_only=True):
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


def main(event_file, midi_file, meta_file, volta_only=True):

    unrolled_event, idx_mapping = unroll_repeat(event_file, volta_only)
    pm, sect_onset = event_to_pm(unrolled_event)

    pm.write(midi_file)

    with open(meta_file, "w") as f:
        json.dump({"idx_mapping": idx_mapping, "onset": sect_onset}, f)

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--event", dest="event_file", type=str,
                        help="Input event file name.")
    parser.add_argument("--meta", dest="meta_file", type=str,
                        help="Output meta file name.")
    parser.add_argument("--midi", dest="midi_file", type=str, default=None,
                        help="Output MIDI file name.")
    parser.add_argument("--audio", dest="audio_file", type=str, default=None,
                        help="Output WAV audio file name.")

    args = parser.parse_args()

    if not args.input:
        raise ValueError("Please specify input file name.")

    if not args.output:
        raise ValueError("Please specify output file name.")

    if args.output.split(".")[-1] != 'wav':
        raise ValueError("Incorrect output format.")

    main(args.event_file, args.meta_file, volta_only=True,
         midi_file=args.midi_file, audio_file=args.audio_file)
