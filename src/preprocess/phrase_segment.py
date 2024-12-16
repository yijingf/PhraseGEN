"""
Segment phrases with predicted boundaries.
"""
import os
import json
import pickle
import numpy as np
from copy import deepcopy
from fractions import Fraction

import sys
sys.path.append("..")

from utils.event import expand_score
from utils.align import get_t_bar, t_to_bar_beat
from utils.common import load_event, trim_event, normalize_event


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
    for _, (_, t_ed) in enumerate(intervals):
        if round(t_ed - last_t_st) < t_thresh:
            continue
        else:
            merged_intervals.append([last_t_st, t_ed])
            last_t_st = t_ed

    if not len(merged_intervals):
        merged_intervals = [intervals[0]]

    if last_t_st != t_ed:
        merged_intervals[-1][-1] = t_ed

    return merged_intervals


def get_t_fin(n_bar, onset, **kwargs):
    """Get time (in seconds) of last bar line.

    Args:
        idx_mapping (dict): _description_
        onset (list): _description_
    """
    n_bar_last_seg = n_bar + 1 - onset[-1]['measure']

    t_last_seg = n_bar_last_seg * get_t_bar(**onset[-1])

    t_fin = onset[-1]['t'] + t_last_seg
    return t_fin


def align_boundary(t_bound, ts_shift, min_phrase=2):

    # Postprocess phrase boundary
    last_idx = -1
    bounds = []
    for i, entry in enumerate(ts_shift[1:]):

        # time signature is consisitent within a phrase
        idx = np.where(t_bound[:, 1] < entry['t'])[0][-1]
        sect_t_bound = np.vstack([t_bound[last_idx + 1: idx + 1],
                                  [t_bound[idx, 1], entry['t']]])

        # merge small segments
        t_thresh = get_t_bar(**ts_shift[i]) * min_phrase
        sect_t_bound = merge_intervals(sect_t_bound, t_thresh=t_thresh)

        # align time with bar-beat
        bounds += [t_to_bar_beat(t[0], ts_shift) for t in sect_t_bound]

        last_idx = idx

    return bounds


def main(event_file, n_cluster=6, min_phrase=2):

    # Load expanded event
    score, struct = load_event(event_file)
    event, idx_mapping = expand_score(score, struct, "no_repeat")

    # Key transpose to C major/minor
    event = normalize_event(event)

    # Load score measure to midi measure mapping
    map_file = event_file.replace(dirConfig.event_dir, dirConfig.midi_dir)
    with open(map_file) as f:
        ts_shift = json.load(f)['onset']

    # Renew end time
    t_fin = get_t_fin(max(idx_mapping), ts_shift)
    ts_shift.append({"measure": max(idx_mapping) + 1, "t": float(t_fin),
                     "tempo": ts_shift[-1]['tempo'],
                     "time_signature": ts_shift[-1]['time_signature']})

    # Load predicted boundaries
    base_name = os.path.basename(event_file).split(".")[0]
    composer = os.path.basename(os.path.pardir(event_file))
    with open(os.path.join(dirConfig.boundary_dir, f"{composer}-{base_name}.pkl"), "rb") as f:
        t_bound = pickle.load(f)[n_cluster][0]
    t_bound[0, 0] = 0
    t_bound[-1, -1] = t_fin

    bounds = align_boundary(t_bound, ts_shift, min_phrase)
    bounds.append((max(idx_mapping) + 1, 0))

    phrases = []
    for i, st in enumerate(bounds[:-1]):
        phrase = trim_event(event, st, bounds[i + 1])
        phrases.append({'time_signature': phrase[min(phrase)]['time_signature'],
                        'tempo': phrase[min(phrase)]['tempo'],
                        'note': [measure['event'] for measure in phrases]})

    with open(os.path.join(dirConfig.phrase_dir, composer, f"{base_name}.json")) as f:
        json.dump(phrases, f)
    return


if __name__ == "__main__":

    from glob import glob
    from utils.constants import DATA_DIR

    class dirConfig:
        event_dir = os.path.join(DATA_DIR, "event")
        midi_dir = os.path.join(DATA_DIR, "rendered_midi_no_repeat")
        phrase_dir = os.path.join(DATA_DIR, "phrase")

    composers = ['scarlatti', 'haydn', 'beethoven', 'mozart']
    for composer in composers:
        for event_file in sorted(
                glob(os.path.join(dirConfig.event_dir, composer, "*.json"))):
            main(event_file)
