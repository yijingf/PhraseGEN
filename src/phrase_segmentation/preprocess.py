# Debug:
# tempo might change within a section

import os
import math
import json
import pickle
import numpy as np

from copy import deepcopy
from fractions import Fraction

import sys
sys.path.append("..")

from kern_utils.common import trim_event, normalize_event
from kern_utils.common import load_event, token2v, get_t_bar
from kern_utils.event import get_sub_sect_event, remove_repeat, no_repeat_pattern, concat_event

DATA_DIR = "../../sonata-dataset"
postfix = ['', '-repeat']


def get_sect_onset_measure(entry_name, sect_interval):

    for entry in sect_interval:
        if entry['name'] == entry_name:
            break
    return entry['measure'][0]


def locate_repeat_sign_in_real_section_A_B(t_sects, sect_interval, mapping):
    t_expose, t_dev, t_recap = t_sects
    # Assume exposure always exists in A

    _, expose_pos = convert_t_to_midi_bar(t_expose, mapping['onset'])

    real_sect_interval = {}

    i_sect_dev, t_dev = locate_section(t_dev, sect_interval)
    expose_sect_interval = [deepcopy(sect_interval[i])
                            for i in range(i_sect_dev)]

    if t_dev > sect_interval[i_sect_dev]['interval'][0]:
        tmp_t = sect_interval[i_sect_dev]["interval"][0]
        expose_sect_interval += [{"name": sect_interval[i_sect_dev]["name"],
                                  "interval": [tmp_t, t_dev],
                                  "measure": convert_t_to_midi_bar(tmp_t,
                                                                   mapping['onset'])}]
    real_sect_interval["expose"] = expose_sect_interval

    real_sect_interval["dev"] = [deepcopy(sect_interval[i_sect_dev])]
    t_fin = deepcopy(sect_interval[i_sect_dev]['interval'])[-1]

    if t_recap:
        i_sect_recap, t_recap = locate_section(t_recap, sect_interval)

        # make sure recap and expose start from the same position in a bar
        i_recap_measure, _ = convert_t_to_midi_bar(t_recap, mapping['onset'])
        t_recap = convert_midi_pos_to_t((i_recap_measure, deepcopy(expose_pos)),
                                        mapping['onset'])

        real_sect_interval["dev"][0]['interval'][-1] = t_recap
        real_sect_interval["recap"] = [{"name": sect_interval[i_sect_recap]["name"],
                                        "interval": [t_recap,
                                                     sect_interval[i_sect_dev]['interval'][-1]],
                                        "measure": [i_recap_measure, deepcopy(expose_pos)]}]

    return real_sect_interval


def locate_repeat_sign_in_real_section(t_sects, sect_interval, mapping):
    t_expose, t_dev, t_recap = t_sects
    # Assume exposure always exists in A

    _, expose_pos = convert_t_to_midi_bar(t_expose, mapping['onset'])

    real_sect_interval = {}
    if t_dev:
        i_sect_dev, t_dev = locate_section(t_dev, sect_interval)
        expose_sect_interval = [deepcopy(sect_interval[i])
                                for i in range(i_sect_dev)]
        if t_dev > sect_interval[i_sect_dev]['interval'][0]:
            tmp_t = sect_interval[i_sect_dev]["interval"][0]
            expose_sect_interval += [{"name": sect_interval[i_sect_dev]["name"],
                                      "interval": [tmp_t, t_dev],
                                      "measure": convert_t_to_midi_bar(tmp_t,
                                                                       mapping['onset'])}]
        real_sect_interval["expose"] = expose_sect_interval

    if t_recap:
        i_sect_recap, t_recap = locate_section(t_recap, sect_interval)

        # make sure recap and expose start from the same position in a bar
        i_recap_measure, _ = convert_t_to_midi_bar(t_recap, mapping['onset'])
        t_recap = convert_midi_pos_to_t((i_recap_measure, deepcopy(expose_pos)),
                                        mapping['onset'])
        t_recap = float(t_recap)

        if t_dev:
            dev_sect_interval = [deepcopy(sect_interval[i])
                                 for i in range(i_sect_dev, i_sect_recap)]

            # if not len(dev_sect_interval):
            #     dev_sect_interval = [{"interval": [t_dev, t_recap]}]
            # else:
            #     dev_sect_interval[0]['interval'][0] = t_dev

            if len(dev_sect_interval):
                dev_sect_interval[0]['interval'][0] = t_dev

            if t_recap > sect_interval[i_sect_recap]['interval'][0]:
                tmp_t = deepcopy(sect_interval[i_sect_recap]["interval"][0])
                dev_sect_interval += [{"name": sect_interval[i_sect_recap]["name"],
                                       "interval": [tmp_t, t_recap],
                                       "measure": convert_t_to_midi_bar(tmp_t,
                                                                        mapping['onset'])}]
            real_sect_interval["dev"] = dev_sect_interval
        else:
            expose_sect_interval = [
                deepcopy(sect_interval[i]) for i in range(i_sect_recap)]
            if t_recap > sect_interval[i_sect_recap]['interval'][0]:
                tmp_t = deepcopy(sect_interval[i_sect_recap]["interval"][0])
                expose_sect_interval += [{"name": sect_interval[i_sect_recap]["name"],
                                          "interval": [tmp_t, t_recap],
                                          "measure": convert_t_to_midi_bar(tmp_t,
                                                                           mapping['onset'])}]
            real_sect_interval["expose"] = expose_sect_interval

        recap_sect_interval = [deepcopy(sect_interval[i])
                               for i in range(i_sect_recap, len(sect_interval))]
        recap_sect_interval[0]['interval'][0] = t_recap
        real_sect_interval["recap"] = recap_sect_interval
        real_sect_interval["recap"][0]["measure"] = convert_t_to_midi_bar(t_recap,
                                                                          mapping['onset'])
    else:
        if t_dev:
            # This is impossible
            pass
        real_sect_interval["expose"] = deepcopy(sect_interval)

    return real_sect_interval


def locate_section(t_sect, sect_interval, t_thresh=5):
    sect_name = [entry['name'] for entry in sect_interval]
    sect_onset = np.array([entry['interval'][0] for entry in sect_interval])
    # Todo: use t_bar for t_thresh
    i_candidates = np.argsort(np.abs(t_sect - sect_onset))

    if sect_onset[i_candidates[0]] <= t_sect:
        i_sect = i_candidates[0]
    elif t_sect - sect_onset[i_candidates[0]] <= t_thresh:
        i_sect = i_candidates[0]
        # revise t_sect if the section boundary is very close to a repeat sign
        t_sect = sect_onset[i_sect]
    else:
        i_sect = i_candidates[:2][-1]

    # If segmentation algorithm failed to identify primary theme in 1st volta
    if 'repeat' in sect_name[i_sect]:
        if i_sect >= 1 and sect_name[i_sect - 1][0] == sect_name[i_sect][0]:
            t_delta = t_sect - sect_onset[i_sect]
            t_sect = sect_onset[i_sect - 1] + t_delta
            i_sect -= 1

    return i_sect, t_sect


def find_sect_A_B(seg_boundaries, seg_cls, sect_interval):
    # Identify primary theme in A
    primary_theme_cls = seg_cls[0]
    t_candidate = seg_boundaries[seg_cls == primary_theme_cls][:, 0]

    # Exposition
    t_expose = t_candidate[0]

    # Development
    for entry in sect_interval:
        if entry['name'] == 'B':
            break
    t_dev = entry['interval'][0]

    # Recapitulation
    if any(t_candidate > t_dev):
        # t_recap = t_candidate[t_candidate > t_dev][0]
        t_recap = t_candidate[t_candidate > t_dev][-1]
    else:
        t_recap = None  # Doesn't follow ABA'

    return t_expose, t_dev, t_recap


def find_sect_A_B_C(seg_boundaries, seg_cls, sect_interval):

    primary_theme_cls = seg_cls[0]
    t_candidate = seg_boundaries[seg_cls == primary_theme_cls][:, 0]

    # Exposition
    t_expose = t_candidate[0]

    i_sect = 0
    recap_sect = None

    recap_sect, t_recap = '', None
    seg_boundaries
    for t in t_candidate[1:]:
        while i_sect < len(sect_interval) and t > sect_interval[i_sect]['interval'][-1]:
            i_sect += 1

        if "A" not in sect_interval[i_sect]["name"]:
            t_recap = t
            recap_sect = sect_interval[i_sect]["name"]
            break

    t_dev = None
    if recap_sect in ['B', 'C']:
        for entry in sect_interval:
            if entry['name'] == 'B':
                t_dev = entry['interval'][0]
                break

    return t_expose, t_dev, t_recap


def find_sect_A_B_A(sect_interval):
    t_expose, t_dev, t_recap = None, None, None

    for entry in sect_interval:
        if entry['name'] == 'A':
            t_expose = entry['interval'][0]
        elif entry['name'] == 'A-recap':
            t_recap = entry['interval'][0]
        elif entry['name'] == 'C':
            t_dev = entry['interval'][0]

    # is_expose = True
    # for entry in sect_interval:
    #     if entry['name'] == 'A':
    #         if is_expose:
    #             t_expose = entry['interval'][0]
    #             is_expose = False
    #         else:
    #             t_recap = entry['interval'][0]
    #             break

    # for entry in sect_interval:
    #     if entry['name'] == 'C':
    #         t_dev = entry['interval'][0]
    #         break

    return t_expose, t_dev, t_recap


def get_pattern_string(merged_pattern):
    pattern = [i[0] for i in merged_pattern]
    pattern_string = []
    last_i = ''
    for i in pattern:
        if i != last_i:
            pattern_string.append(i)

        last_i = i
    return '-'.join(pattern_string)


def merge_sub_sect(sub_sects):
    """Merge sub-section with same prefix, such as A, A1, A, A2 into one section [A, A1], [A, A2]

    Args:
        sub_sects (list): _description_

    Returns:
        _type_: _description_
    """
    sect, sects = [], []
    last_sub_sect = sub_sects[0]

    for sub_sect in sub_sects:

        if sub_sect[0] != last_sub_sect[0] or len(sub_sect) < len(last_sub_sect):
            sects.append(sect)
            sect = [sub_sect]
        else:
            sect.append(sub_sect)
        last_sub_sect = sub_sect

    if sect:
        sects.append(sect)

    return sects


def get_sect_duration(events):
    i_st, i_ed = 0, max(events)

    if len(events[i_st]['event']):
        assert events[i_st]['event'][0][0] == 'o'

    numerator = Fraction(events[i_st]['time_signature']) * 4
    if len(events[i_st]['event']):
        onset_pos = token2v(events[i_st]['event'][0])
    else:
        onset_pos = numerator
    t_bar = get_t_bar(**events[i_st])
    t_onset = onset_pos / numerator * t_bar

    t_sect = 0
    for i in range(i_st, i_ed + 1):
        t_sect += get_t_bar(**events[i])

    return t_sect, t_onset, get_t_bar(**events[i_ed])


def get_sect_interval(sect_duration):
    sect_name_set = set()
    sect_interval = []
    for i, entry in enumerate(sect_duration):
        entry_name = entry['name']
        if entry_name not in sect_name_set:
            sect_name_set.add(entry_name)
        else:
            entry_name = f"{entry_name}-recap"
            sect_name_set.add(entry_name)

        delta = 0
        if i < len(sect_duration) - 1:
            next_onset, next_t_bar = sect_duration[i + 1]['duration'][1:]
            if next_onset:
                delta = next_t_bar - next_onset
            else:
                delta = 0

        if i < 1:
            t_st = 0
            t_ed = t_st + entry['duration'][0] - delta
        else:
            t_st = sect_interval[i - 1]['interval'][1]
            t_ed = t_st + entry['duration'][0] - entry['duration'][1] - delta

        sect_interval.append({"name": entry_name,
                              "interval": [float(t_st), float(t_ed)]})
    return sect_interval


def find_closest_t_section(i_measure, onsets):
    """not necessarily the Section/Pattern in Kern

    Args:
        t (_type_): _description_
    """

    # Find closest tempo/time signature changing point
    for i, entry in enumerate(onsets):
        if i < len(onsets) - 1:
            if i_measure < onsets[i + 1]['measure']:
                break
        else:
            break

    sect_measure = entry['measure']
    _, sect_pos = convert_t_to_midi_bar(entry['t'], onsets)
    t_bar = get_t_bar(**entry)
    ts = standard_ts(entry['time_signature'])
    numerator = int(Fraction(ts) * 4)

    # NOTE: Heuristic: estimated primary theme boundary fall in the correct measure
    t_diff = (i_measure - sect_measure + sect_pos / numerator) * t_bar

    return entry['t'] + t_diff


def convert_midi_pos_to_t(pos, onsets):
    """not necessarily the Section/Pattern in Kern

    Args:
        pos: (i_measure, pos)
        t (_type_): _description_
    """
    i_measure, onset_pos = pos

    # Find closest tempo/time signature changing point
    for i, entry in enumerate(onsets):
        if i < len(onsets) - 1:
            if i_measure < onsets[i + 1]['measure']:
                break
        else:
            break

    sect_measure = entry['measure']
    _, sect_pos = convert_t_to_midi_bar(entry['t'], onsets)
    t_bar = get_t_bar(**entry)
    ts = standard_ts(entry['time_signature'])
    numerator = int(Fraction(ts) * 4)

    # NOTE: Heuristic: estimated primary theme boundary fall in the correct measure
    t_diff = (i_measure - sect_measure + sect_pos / numerator) * t_bar

    t_diff += onset_pos / numerator * t_bar

    return entry['t'] + t_diff


def get_structure(seg_boundaries, seg_cls, sect_interval, mapping, struct):
    """Identify exposition, development, and recapitulation
    """

    # Identify primary theme
    # Assume notation should start from A, all pieces has an A
    i = None
    for i, entry in enumerate(sect_interval):
        if entry['name'][0] == 'A':
            break

    # Exposition
    # Assume that the first segment in exposition is the primary theme
    i_seg = np.argmin(
        np.abs(seg_boundaries[:, 0] - sect_interval[i]['interval'][0]))
    boundary_i = [i for i, v in enumerate(seg_cls)
                  if v == seg_cls[i_seg] and i >= i_seg]
    primary_theme_t = seg_boundaries[boundary_i][:, 0]

    t_expose = 0
    for i, entry in enumerate(sect_interval):
        if entry['name'][0] == 'A':
            t_expose = entry['interval'][0]
        break

    expose_i_measure, _ = convert_t_to_midi_bar(t_expose,
                                                mapping['onset'])
    kern_expose_i_measure = token2v(struct['attr']['A']['idx'])
    pos = token2v(struct['attr']['A']['onset'])

    # Recapitulation
    # Find primary theme in recapitulation
    t_recap = None
    for t in primary_theme_t[1:]:
        i_sect = 0
        while i_sect < len(sect_interval) - 1 and t > sect_interval[i_sect]['interval'][1]:
            i_sect += 1

        # Assume that the primary theme in the first section other than A marks the recapitulation
        if sect_interval[i_sect]["name"][0] != "A":
            t_recap = t
            break

    if not t_recap:
        pass  # Todo: Error handling

    # Assume the recapitulation starting from the same position as exposition
    recap_i_measure, _ = convert_t_to_midi_bar(t_recap,
                                               mapping['onset'])
    normed_t_recap = find_closest_t_section(recap_i_measure, mapping['onset'])
    kern_recap_i_measure = mapping['idx_mapping'][str(recap_i_measure)]

    res = {"Exposition": {"t": t_expose,
                          "measure": expose_i_measure,  # midi measure
                          "pos": pos},
           "Recapitulation": {"t": normed_t_recap,
                              "measure": recap_i_measure,
                              "pos": pos}
           }

    return


def standard_ts(ts):
    if len(ts.split("/")) > 2:
        a, b, c = ts.split('/')
        new_b = str(int(b) * int(c))
        ts = f"{a}/{new_b}"
    return ts


def convert_t_to_midi_bar(t, sect_onsets, scale=6):
    # Bar index in unrolled score/MIDI
    n_shifts = len(sect_onsets)
    if t >= sect_onsets[-1]['t']:
        i = n_shifts
    else:
        for i in range(1, n_shifts):
            if t <= sect_onsets[i]['t']:
                break

    i -= 1
    tp = sect_onsets[i]['tempo']
    ts = sect_onsets[i]['time_signature']
    std_ts = standard_ts(ts)

    numerator = int(Fraction(std_ts) * 4)
    t_bar = get_t_bar(ts, tp)

    t_sect_onset = sect_onsets[i]['t']

    midi_measure_idx = sect_onsets[i]['measure']
    # round to closest measure
    round_thresh = (numerator - 1 / scale) / numerator
    n_measure = math.ceil((t - t_sect_onset) / t_bar - round_thresh)
    t_pos = t - t_sect_onset - t_bar * n_measure
    midi_measure_idx += n_measure

    bins = np.arange(0, numerator, 1 / scale)
    pos = Fraction(np.argmin(np.abs(t_pos / t_bar * 4 - bins)), scale)

    return int(midi_measure_idx), pos


def get_t_fin(idx_mapping, onset, **kwargs):
    """Get time (in seconds) of last bar line.

    Args:
        idx_mapping (dict): _description_
        onset (list): _description_
    """
    n_bar = max([int(i) for i in idx_mapping.keys()]) + 1
    n_bar_last_seg = n_bar - onset[-1]['measure']

    t_last_seg = n_bar_last_seg * get_t_bar(**onset[-1])

    t_fin = onset[-1]['t'] + t_last_seg
    return t_fin


def merge_intervals(intervals, t_thresh=1):
    """Merge intervals shorter than `t_thresh` into longer phrases.

    Args:
        intervals (list): A list of phrase start, ending time, i.e. [[t_start, t_end]]
        t_thresh (int, optional): Defaults to 1.

    Returns:
        _type_: _description_
    """
    merged_intervals = []
    last_t_st = intervals[0][0]
    for _, (t_st, t_ed) in enumerate(intervals):
        if t_ed - t_st <= t_thresh:
            continue
        else:
            merged_intervals.append([last_t_st, t_ed])
            last_t_st = t_ed

    if not len(merged_intervals):
        merged_intervals = [intervals[0]]

    if last_t_st != t_ed:
        merged_intervals[-1][-1] = t_ed

    return merged_intervals


def retrieve_section_interval(sub_sect_event, sub_sects, t_fin):
    """Retrieve start/ending time of each section on score.

    Args:
        sub_sect_event (dict): _description_
        sects (): _description_
        t_fin (float): time of last bar line.

    Returns:
        sub_sect_interval (list): a list of section attributes, i.e. 
            [{"name": `Section Name`, 
              "interval": [`Section Start Time`, `Section End Time`]}]
    """

    i_measure = 0
    sub_sect_onset = []
    t_last_onset = 0
    t_onset = 0

    for i_sect, sub_sect in enumerate(sub_sects):

        note_event = sub_sect_event[sub_sect]
        i_st = min(note_event)
        i_ed = max(note_event)

        pos_onset_token = sub_sect_event[sub_sect][i_st]['event'][0]
        pos_onset = token2v(pos_onset_token)

        entry = {"name": sub_sect, "onset": [i_measure, pos_onset]}

        if i_sect < len(sub_sects) - 1:
            i_measure += i_ed - i_st

        curr_tempo = int(note_event[i_st]['tempo'] / 12) * 12  # Normalized!
        t_curr_onset = float(pos_onset) * 60 / curr_tempo
        t_onset += t_curr_onset

        if i_sect:
            prev_sect_event = sub_sect_event[sub_sects[i_sect - 1]]
            prev_i_st = min(prev_sect_event)
            prev_i_ed = max(prev_sect_event)
            t_bar = get_t_bar(**list(prev_sect_event.values())[-1])
            t_onset += (prev_i_ed - prev_i_st + 1) * t_bar - t_last_onset

        entry['time'] = t_onset
        sub_sect_onset.append(entry)
        t_last_onset = t_curr_onset

    sub_sect_interval = []
    for i, entry in enumerate(sub_sect_onset):
        t_st = entry['time']
        if i + 1 < len(sub_sect_onset):
            t_ed = sub_sect_onset[i + 1]['time']
        else:
            t_ed = t_fin
        sub_sect_interval.append({"name": entry['name'],
                                  "interval": [t_st, t_ed]})
    return sub_sect_interval


def locate_phrase(sect_interval, boundaries, t_thresh=1):
    """Locate the phrases in sections given phrase start/ending time. Modify `sect_interval` in place.

    Args:
        sect_interval (list): a list of section attributes, i.e. 
            [{"name": `Section Name`, 
              "interval": [`Section Start Time`, `Section End Time`]}]
        boundaries (_type_): _description_

    Returns:
        _type_: _description_
    """

    i_sect = 0
    for t_st, t_ed in boundaries:
        if t_ed < sect_interval[i_sect]['interval'][0]:
            continue

        t_sect_ed = sect_interval[i_sect]['interval'][-1]

        if t_st < t_sect_ed:
            if t_ed <= t_sect_ed:
                sect_interval[i_sect]['phrase'] = sect_interval[i_sect].get(
                    'phrase', []) + [[t_st, t_ed]]
            else:
                # Cut off the phrase with current section ending
                if i_sect < len(sect_interval) - 1:
                    sect_interval[i_sect]['phrase'] = sect_interval[i_sect].get(
                        'phrase', []) + [[t_st, t_sect_ed]]
                    i_sect += 1
                    sect_interval[i_sect]['phrase'] = sect_interval[i_sect].get(
                        'phrase', []) + [[t_sect_ed, t_ed]]
                else:
                    sect_interval[i_sect]['phrase'] = sect_interval[i_sect].get(
                        'phrase', []) + [[t_st, t_ed]]

        # Shift to next section
        elif i_sect < len(sect_interval) - 1:
            i_sect += 1
            t_sect_ed = sect_interval[i_sect]['interval'][-1]
            while t_ed > t_sect_ed and i_sect < len(sect_interval) - 1:
                sect_interval[i_sect]['phrase'] = sect_interval[i_sect].get(
                    'phrase', []) + [[t_st, t_sect_ed]]
                t_st = t_sect_ed
                i_sect += 1
                t_sect_ed = sect_interval[i_sect]['interval'][-1]
            sect_interval[i_sect]['phrase'] = sect_interval[i_sect].get(
                'phrase', []) + [[t_st, t_ed]]

    # Postprocess, merge short phrases into longer ones
    for i_sect, entry in enumerate(sect_interval):
        sect_interval[i_sect]['phrase'][0][0] = entry['interval'][0]
        sect_interval[i_sect]['phrase'][-1][-1] = entry['interval'][-1]
        sect_interval[i_sect]['phrase'] = merge_intervals(entry['phrase'],
                                                          t_thresh)

    return sect_interval


def retrieve_phrase_event(composer, file_base, phrase_level=5, seg_level=-1, mode='no_repeat'):

    # Load events
    event_file = os.path.join(DATA_DIR, "event", composer, file_base)
    event, struct = load_event(event_file)

    # Load score measure to midi measure mapping
    if mode:
        mapping_file = os.path.join(DATA_DIR,
                                    f"rendered_midi_{mode}", composer, file_base)
    else:
        mapping_file = os.path.join(DATA_DIR,
                                    "rendered_midi", composer, file_base)

    with open(mapping_file) as f:
        mapping = json.load(f)

    # Load predicted boundaries
    name = file_base.split(".")[0]
    boundary_file_name = os.path.join(DATA_DIR,
                                      "boundary_predictions",
                                      f"{composer}-{name}.pkl")
    with open(boundary_file_name, "rb") as f:
        boundaries_sets = pickle.load(f)

    phrase_boundaries, _ = boundaries_sets[phrase_level]
    seg_boundaries, seg_cls = boundaries_sets[seg_level]
    seg_cls = np.array(seg_cls)

    # Mapping phrase boundaries in time to bar-note annotation
    t_fin = get_t_fin(**mapping)
    t_onsets = [i['t'] for i in mapping['onset']]
    t_onsets.append(t_fin)

    # Retrieve note events for each section
    onsets = sorted([(i, v) for i, v in struct['attr'].items()],
                    key=lambda x: (x[1]['idx'], x[1]['onset']))
    sub_sect_event = get_sub_sect_event(event, onsets)

    # Retrieve section start/ending time
    sub_sects = struct['pattern']
    # sub_sects = remove_repeat(sub_sects)
    sub_sects = no_repeat_pattern(sub_sects)

    # Merge Sub-section on Score
    sects = merge_sub_sect(sub_sects)

    last_sect_name = ''
    # i_sub_sect, sect_idx = 0, 0
    sect_idx = 0
    sect_event = {}
    sect_duration = []

    # Sections on Score
    for sect in sects:

        sect_name = sect[0][0]
        if sect_name != last_sect_name:
            sect_idx = 0
        else:
            sect_idx += 1
        last_sect_name = sect_name

        tmp_events = concat_event(sub_sect_event, sect, struct['attr'])[0]
        sect_event[f'{sect_name}{postfix[sect_idx]}'] = tmp_events

        sect_duration.append({"name": f'{sect_name}{postfix[sect_idx]}',
                              "duration": get_sect_duration(tmp_events)})

    sect_interval = get_sect_interval(sect_duration)
    for i, entry in enumerate(sect_interval):
        sect_interval[i]['measure'] = convert_t_to_midi_bar(
            entry['interval'][0], mapping['onset'])

    # Get sections based on phrase segmentation
    pattern_str = get_pattern_string(sects)

    if pattern_str[0] != 'I':

        if pattern_str == 'A':
            # Use first seg to mark section
            t_sects = np.append(seg_boundaries[np.where(
                seg_cls == seg_cls[0])][:, 0], t_fin)

            # normalize t
            # sect_measure = [t_sects[0]] + [convert_t_to_midi_bar(t, mapping['onset'])[0] for i in t_sects[1:]]
            # t_sects =[t_sects[0]] + [find_closest_t_section(i, mapping['onset']) for i in sect_measure]

            real_sect_interval = {"expose": [{"name": f"A-{i}",
                                              "interval": [v, t_sects[i + 1]],
                                              "measure": (0, 0)}
                                             for i, v in enumerate(t_sects[:-1])]}

        elif pattern_str == 'A-B':
            t_sects = find_sect_A_B(seg_boundaries, seg_cls, sect_interval)
            real_sect_interval = locate_repeat_sign_in_real_section_A_B(
                t_sects, sect_interval, mapping)

        elif len(pattern_str.split('A-B')) > 2:
            # AB-x-AB or AB-x-AB-y
            t_sects = find_sect_A_B_A(sect_interval)
            real_sect_interval = locate_repeat_sign_in_real_section(
                t_sects, sect_interval, mapping)

            if pattern_str[-1] == 'B':
                # AB-x-AB Type, remove recap
                del real_sect_interval['recap']
        else:
            # A-B-A' type
            t_sects = find_sect_A_B_C(seg_boundaries, seg_cls, sect_interval)
            real_sect_interval = locate_repeat_sign_in_real_section(
                t_sects, sect_interval, mapping)
    else:
        # if Long I: treat as A-type
        if struct['attr']['A']['idx'] - struct['attr']['I']['idx'] >= 4:
            real_sect_interval = {"expose": deepcopy(sect_interval)}
        else:
            return None

    # # Locate phrases in segmentation
    for sect_name, sect in real_sect_interval.items():
        # Modify in place
        real_sect_interval[sect_name] = locate_phrase(
            deepcopy(sect), phrase_boundaries, t_thresh=5)

    # Make sure repeated section have same segmentation
    # NOTICE: Not neccesary! because boundaries are calculated on no_repeat ver. now

    # for sect_name, sect in real_sect_interval.items():
    #     for i, entry in enumerate(sect):
    #         if "repeat" in entry['name']:
    #             if i > 0 and entry['name'][0] == sect[i - 1]['name'][0]:
    #                 phrase = np.array(
    #                     sect[i - 1]['phrase']) - sect[i - 1]['interval'][0] + entry['interval'][0]
    #                 entry['phrase'] = phrase.tolist()
    #         else:
    #             continue

    # Convert time to bar-position
    for sect_name, sect in real_sect_interval.items():
        for entry in sect:
            measure_onset = get_sect_onset_measure(entry['name'],
                                                   sect_interval)
            entry['phrase_measure'] = []
            for phrase in entry['phrase']:
                measure, pos = convert_t_to_midi_bar(phrase[0],
                                                     mapping['onset'])
                # measure -= entry['measure'][0]
                measure -= measure_onset
                entry['phrase_measure'].append([measure, pos])

            measure, pos = convert_t_to_midi_bar(entry['interval'][1],
                                                 mapping['onset'])
            # measure -= entry['measure'][0]
            measure -= measure_onset
            entry['phrase_measure'].append([measure, pos])

    # Get phrase event and phrase pair index
    sect_phrase = {}

    for sect_name, sect in real_sect_interval.items():
        sect_phrase[sect_name] = {}

        sect_phrase[sect_name]['phrase'] = []
        sect_phrase[sect_name]['pair'] = []

        i_phrase = 0
        for entry in sect:
            for i, st in enumerate(entry['phrase_measure'][:-1]):
                score_sect_name = entry["name"].split("-")[0]
                phrase_event = trim_event(sect_event[score_sect_name],
                                          st,
                                          entry['phrase_measure'][i + 1])
                if not len(phrase_event):
                    continue
                phrase_event = normalize_event(phrase_event)
                i_st, i_ed = min(phrase_event), max(phrase_event)
                note_events = [phrase_event[i_measure]['event']
                               for i_measure in range(i_st, i_ed + 1)]
                sect_phrase[sect_name]['phrase'].append({"tempo": phrase_event[i_st]["tempo"],
                                                         "time_signature": phrase_event[i_st]["time_signature"],
                                                         "event": note_events})

                if i < len(entry['phrase_measure']) - 2:
                    sect_phrase[sect_name]['pair'].append(
                        [i_phrase, i_phrase + 1])

                i_phrase += 1

    return sect_phrase

# def retrieve_phrase_event(composer, file_base, phrase_level=5, seg_level=-1, volta_only=True, t_phrase_thresh=1):

#     # Load events
#     event_file = os.path.join(DATA_DIR, "event", composer, file_base)
#     event, struct = load_event(event_file)

#     # Load score measure to midi measure mapping
#     with open(os.path.join(DATA_DIR, "rendered_midi", composer, file_base)) as f:
#         mapping = json.load(f)

#     # Load predicted boundaries
#     name = file_base.split(".")[0]
#     boundary_file_name = os.path.join(DATA_DIR,
#                                       "boundary_predictions",
#                                       f"{composer}-{name}.pkl")
#     with open(boundary_file_name, "rb") as f:
#         boundaries_sets = pickle.load(f)
#     phrase_boundaries, _ = boundaries_sets[phrase_level]
#     seg_boundaries, seg_cls = boundaries_sets[seg_level]

#     # Mapping phrase boundaries in time to bar-note annotation
#     t_fin = get_t_fin(**mapping)
#     t_onsets = [i['t'] for i in mapping['onset']]
#     t_onsets.append(t_fin)

#     t_barlines = []
#     for i, entry in enumerate(mapping['onset']):
#         t_st, t_ed = entry['t'], t_onsets[i + 1]
#         t_bar = get_t_bar(**entry)
#         t_barlines.append(np.arange(t_st, t_ed, t_bar) + t_bar)
#     t_barlines = np.concatenate(t_barlines)

#     # Retrieve note events for each section
#     onsets = sorted([(i, v) for i, v in struct['attr'].items()],
#                     key=lambda x: (x[1]['idx'], x[1]['onset']))
#     sub_sect_event = get_sub_sect_event(event, onsets)

#     # Retrieve section start/ending time
#     sects = struct['pattern']
#     if volta_only:
#         sects = remove_repeat(sects)
#     sect_interval = retrieve_section_interval(sub_sect_event, sects, t_fin)

#     # Locat phrases in sections on score, modify `sect_interval` in place
#     sect_interval = locate_phrase(sect_interval,
#                                   phrase_boundaries,
#                                   t_phrase_thresh)

#     # Retrieve note events for each phrase.
#     sect_phrase_events = []
#     for entry in sect_interval:

#         # Note events within the current sub section
#         note_event = sub_sect_event[entry['name']]
#         # Min/Max measure index of the current section on score
#         i_st, i_ed = min(note_event), max(note_event)

#         t_bar = get_t_bar(**note_event[i_st])
#         beat_per_bar = t_bar * int(note_event[i_st]['tempo']) / 60

#         i_measure = i_st
#         i_token = 0
#         t_st = entry['interval'][0]

#         phrase_events = []  # phrases in current section
#         for _, t_ed_phrase in entry['phrase']:

#             phrase_entry = {"tempo": note_event[i_st]['tempo'],
#                             "key": note_event[i_st]['key'],
#                             "time_signature": note_event[i_st]['time_signature']}

#             phrase_event = []

#             if i_token:
#                 phrase_event += note_event[i_measure]['event'][i_token:] + ['bar']
#                 i_measure += 1

#             while i_measure < i_ed and (i_measure - i_st) * t_bar + t_st <= t_ed_phrase:
#                 phrase_event += note_event[i_measure]['event']
#                 phrase_event += ['bar']
#                 i_measure += 1

#             reset_bar = True
#             for i_token, token in enumerate(note_event[i_measure]['event']):
#                 if token[0] != 'o':
#                     phrase_event.append(token)
#                 elif (token2v(token) / beat_per_bar + i_measure - i_st) * t_bar > t_ed_phrase:
#                     reset_bar = False
#                     break
#                 else:
#                     phrase_event.append(token)

#             if reset_bar:
#                 i_token = 0
#                 i_measure += 1

#             phrase_entry['event'] = phrase_event
#             phrase_events.append(phrase_entry)

#         sect_phrase_events.append(
#             {"name": entry["name"], "phrase_events": phrase_events})

#     return sect_phrase_events


if __name__ == "__main__":
    pass

    # composers = os.listdir(os.path.join(DATA_DIR, "event"))

    # for composer in composers:
    #     for event_file in sorted(os.listdir(os.path.join(DATA_DIR, "event", composer))):
    #         event, struct = load_event(os.path.join(
    #             DATA_DIR, "event", composer, event_file))
    #         sects = remove_repeat(struct['pattern'])
    #         print(composer, event_file)
    #         print(sects)
