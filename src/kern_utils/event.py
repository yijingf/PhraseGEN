import numpy as np
from copy import deepcopy

from common import trim_event, token2v


def remove_repeat(pattern):
    new_pattern = []
    last_sect = ''
    for sect in pattern:
        if sect != last_sect:
            new_pattern.append(sect)
        last_sect = sect
    return new_pattern


def concat_measure(phrase, add_eos=True, eos_token='eos',
                   bar_eos_token='sep', pad_bar=True, max_measure_len=64):

    tokens = [phrase['time_signature'], phrase['tempo']]

    n_measure = len(phrase['note'])

    if pad_bar:
        tokens += phrase['note'][0] + ['bar']
        measure_len = [0, len(phrase['note'][0]) + 1]

        for i in range(1, n_measure - 1):
            notes = phrase['note'][i]

            pad_len = max_measure_len - len(notes)
            if len(notes) < max_measure_len:
                notes += [bar_eos_token for _ in range(pad_len)]

            tokens += notes
            tokens += ['bar']
            measure_len += [len(notes) + 1]

        tokens += phrase['note'][-1]
        tokens += [eos_token]
        measure_len += [len(phrase['note'][-1]) + 1]

    else:
        tokens += [token for bar in phrase['note'] for token in bar + ['bar']]
        tokens[-1] = eos_token

        measure_len = np.array([0] + [len(measure)
                               for measure in phrase['note']])
        measure_len += np.ones(n_measure + 1, dtype=int)
        measure_len[0] = 0

    idx = np.cumsum(measure_len)
    idx += 2

    if not add_eos:
        tokens = tokens[:-1]

    return tokens, idx


def mask_measure_to_idx(bar_idx, mask_measure):
    """_summary_

    Args:
        bar_idx (_type_): _description_
        mask_measure (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not len(mask_measure):
        return []

    if isinstance(mask_measure, list):
        mask_measure = np.array(sorted(mask_measure))

    st_idx = bar_idx[mask_measure]
    ed_idx = bar_idx[mask_measure + 1]
    mask_idx = [j for i in range(len(mask_measure))
                for j in list(range(st_idx[i], ed_idx[i]))]

    return mask_idx


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
