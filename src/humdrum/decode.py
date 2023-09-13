import warnings
import pretty_midi
from fractions import Fraction

from common import token2v

default_time_signature = '4/4'
default_tempo = 120
default_velocity = 100


def pitch_name_to_pm_pitch(pitch_name):

    if '-' in pitch_name:
        pitch_name = "".join(pitch_name.split('-'))
        pitch = pretty_midi.note_name_to_number(pitch_name) - 1
    else:
        pitch = pretty_midi.note_name_to_number(pitch_name)

    return pitch


def decode_time_signature(ts_str):
    ts_digit = ts_str.split('/')
    if len(ts_digit) > 2:
        ts_num = Fraction(int(ts_digit[0]), int(ts_digit[1]))
    else:
        ts_num = int(ts_digit[0])
    ts_denom = int(ts_digit[-1])
    return ts_num, ts_denom


def decode_ts_tp(tokens):

    ts, tp = None, None

    for token in tokens:

        if token[:2] == 'ts':
            ts = decode_time_signature(token[3:])
        elif token[:2] == 'tp':
            tp = int(token[3:])

        if ts and tp:
            break

    return ts, tp


def decode_token_to_event(tokens, bar_eos_token='sep'):
    ts, tp = decode_ts_tp(tokens)
    ts_num, _ = ts

    events = {}
    notes = []

    onset = None
    bar = 0
    events[bar] = []

    seq_len = len(tokens)

    i = 0
    decode_flag = True

    while i < seq_len:
        token = tokens[i]
        i += 1

        if token == 'bar':
            decode_flag = True
            onset = Fraction(0)

            for note in notes:
                dur = ts_num - note[0]
                events[bar].append(note + [dur])

            bar += 1
            events[bar] = []
            notes = []
            continue

        if not decode_flag:
            continue

        if token == bar_eos_token:
            decode_flag = False
            continue

        elif token[:2] in ['ts', 'tp']:
            continue

        elif token[0] == 'o':
            onset = token2v(token)

        elif token[0] == 'd':
            dur = token2v(token)
            for note in notes:
                events[bar].append(note + [dur])

            notes = []

        else:
            notes.append([onset, token])

    return ts, tp, events


def decode_token_to_pm(tokens, bar_eos_token='sep', bar_limit=False, t_offset=0):

    ts, tp, events = decode_token_to_event(tokens, bar_eos_token)

    if ts is None:
        warnings.warn(f"No time signature. Set to {default_time_signature}")
        ts_frac = Fraction(default_time_signature)
        ts = (ts_frac.numerator, ts_frac.denominator)

    if not tp:
        warnings.warn(f"No tempo. Set to {default_tempo}")
        tp = default_tempo

    t_quarter_note = 60 / tp

    ts_num, ts_denom = ts

    assert ts_denom == 4

    inst = pretty_midi.Instrument(program=0)
    n_bar = len(events)
    for i in range(n_bar):
        for onset, pitch_name, duration in events[i]:

            if bar_limit and onset >= ts_num:
                continue

            t_st = (onset + i * ts_num) * t_quarter_note + t_offset
            t_ed = t_st + duration * t_quarter_note
            pitch = pitch_name_to_pm_pitch(pitch_name)
            note = pretty_midi.Note(default_velocity, pitch, t_st, t_ed)
            inst.notes.append(note)

    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(inst)
    return pm
