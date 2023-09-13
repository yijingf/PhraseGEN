import json
import numpy as np
from copy import deepcopy
from music21 import pitch
from fractions import Fraction

# Build key transpose mapping
# C#4 to G4 -> C4; G#3 to B3 -> C4
pitch_offset_dict = {}

base_ps = pitch.Pitch('C4').ps
pitch_pivot = base_ps + 12 / 2

for key in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
    for acc in ['', '-', '#']:

        ks = f"{key}{acc}"

        ks_ps = pitch.Pitch(ks).ps
        if ks_ps > pitch_pivot:
            ks_ps -= 12

        pitch_offset_dict[ks] = base_ps - ks_ps

# Bins of regular tempi
tempo_bin = np.array([24, 40, 60, 72, 96, 120, 144, 160, 192, 200])


def token2v(token):
    """Convert onset/duration token to value.
    Args:
        token (str): _description_

    Returns:
        Value as fraction.
    """
    return Fraction(token.split('-')[-1])


def ts_tp_ratio(ts, tp):
    ratio = 1
    ts_denom = int(ts.split("/")[-1])
    if tp <= 72 and ts_denom == 8:
        ratio = Fraction(2, 1)

    elif tp >= 192 and ts_denom == 2:
        ratio = Fraction(1, 2)

    return ratio


def normalize_tp(tp):
    """Normalize tempo to its closest regular tempo.

    Args:
        tp (int): original tempo

    Returns:
        int: normalized tempo
    """
    idx = np.argmin(np.abs(tp - tempo_bin))
    return int(tempo_bin[idx])


def normalize_ts(ts, base=4):
    """Normalize time signature with denominator of 4.

    Args:
        ts (Fraction): Time signature as fraction.
        base (int, optional): Denominator of the time signature. Defaults to 4.

    Returns:
        str: normalized time signature string with a denominator of 4
    """
    if isinstance(ts, str):
        ts = Fraction(ts)
    ts_num = ts.numerator * Fraction(base, ts.denominator)
    normed_ts = f"{ts_num}/{base}"
    return normed_ts


def normalize_ts_tp(ts, tp):
    ratio = ts_tp_ratio(ts, tp)
    ts = Fraction(ts) * ratio
    normed_ts = normalize_ts(ts)
    normed_tp = normalize_tp(tp * ratio)
    return normed_ts, normed_tp


def pitch_transpose(pitch_token, offset):
    note_ps = pitch.Pitch(pitch_token).ps + offset
    return pitch.Pitch(note_ps).nameWithOctave


def time_transpose(token, ratio=1):

    token_type, v = token.split('-')
    t = Fraction(v) * ratio

    return f"{token_type}-{t}"


def normalize_event(event):

    for measure in event.values():
        pitch_offset = pitch_offset_dict[measure['key'].split()[0]]
        ratio = ts_tp_ratio(measure['time_signature'], measure['tempo'])

        for i, token in enumerate(measure['event']):
            if token[0] in ['o', 'd']:
                if ratio != 1:
                    measure['event'][i] = time_transpose(token, ratio)
            else:
                if pitch_offset:
                    measure['event'][i] = pitch_transpose(token, pitch_offset)

    return event


def load_event(fname):
    with open(fname) as f:
        event = json.load(f)

    note_event = {}
    for i in event['note']:
        note_event[int(i)] = event['note'][i].copy()

    struct = event['struct']

    return note_event, struct


def remove_repeat(pattern):
    new_pattern = []
    last_sect = ''
    for sect in pattern:
        if sect != last_sect:
            new_pattern.append(sect)
        last_sect = sect
    return new_pattern


def trim_event(measures, start=(0, 0), end=(0, 0)):
    """Trim a list of measures given the start/end time.

    Args:
        event (list): _description_
        start (tuple, optional): (measure index, beat/quarter note within a measure). Defaults to (0, 0).
        end (tuple, optional): _description_. Defaults to (0, 0).

    Returns:
        _type_: _description_
    """
    i_st, offset_st = start
    i_ed, offset_ed = end

    seg_measures = {}
    for i_measure in range(i_st, i_ed):
        seg_measures[i_measure] = deepcopy(measures[i_measure])

    # Add notes from last measure
    if offset_ed > 0:
        for i_token, token in enumerate(measures[i_ed]['event']):
            if token[0] == 'o':
                if token2v(token) >= offset_ed:
                    break

        seg_measures[i_ed] = deepcopy(measures[i_ed])
        seg_measures[i_ed]['event'] = seg_measures[i_ed]['event'][:i_token]

    # Remove redundant notes from the first measure
    if offset_st > 0:
        for i_token, token in enumerate(measures[i_st]['event']):
            if token[0] == 'o':
                if token2v(token) >= offset_st:
                    break

        seg_measures[i_st]['event'] = seg_measures[i_st]['event'][i_token:]
    return seg_measures


def concat_event(sub_sect_event, sects, sect_onset_dict, i_measure=0):
    """Concatenate events from subsections.

    Args:
        sub_sect_event (_type_): _description_
        sects (_type_): _description_
        sect_onset_dict (dict): A dictionary of section onset, e.g. `{"A": {"idx": 0, "pos": "o-0}}`.
        i_measure (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    res = {}
    idx_mapping = {}

    for i_sect, sub_sect in enumerate(sects):

        event = sub_sect_event[sub_sect]
        i_st = min(event)
        i_ed = max(event)

        for i in range(i_st, i_ed + 1):
            if i_measure in res:
                res[i_measure]['event'] += deepcopy(event[i]['event'])
            else:
                idx_mapping[i_measure] = i
                res[i_measure] = deepcopy(event[i])
            i_measure += 1

        if i_sect < len(sects) - 1:
            next_sub_sect = sects[i_sect + 1]
            if sect_onset_dict[next_sub_sect]['onset'] != 'o-0':
                i_measure -= 1

    return res, idx_mapping


def get_sub_sect_event(event, sub_sect_onset):
    sub_sect_onset.append(('Fin', {"idx": max(event) + 1,
                                   "onset": "o-0"}))

    # Get events for each sub section
    sub_sect_event = {}
    for i, v in enumerate(sub_sect_onset[:-1]):
        sub_sect = v[0]

        i_st = v[1]['idx']
        offset_st = token2v(v[1]['onset'])

        i_ed = sub_sect_onset[i + 1][1]['idx']
        offset_ed = token2v(sub_sect_onset[i + 1][1]['onset'])

        sub_sect_event[sub_sect] = trim_event(event,
                                              start=(i_st, offset_st),
                                              end=(i_ed, offset_ed))

    return sub_sect_event
