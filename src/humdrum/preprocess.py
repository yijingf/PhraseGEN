import os
import json
import random
import numpy as np
from music21 import pitch
from fractions import Fraction


# Build key transpose mapping
# C#4 - G4 -> C4; G#3 - B3 -> C4
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

# Regular tempo
tempo_bin = np.array([24, 40, 60, 72, 96, 120, 144, 160, 192, 200])


def has_irregular_token(tokens, vocab):
    return any([token not in vocab for token in tokens])


def get_time(event):
    return Fraction(event.split('-')[-1])


def token2v(token):
    """Token to value

    Args:
        token (_type_): _description_
        prefix (str, optional): _description_. Defaults to 'o'.

    Returns:
        _type_: _description_
    """
    return Fraction(token.split('-')[-1])


def norm_tp(tp):
    """Normalize tempo to its closets regular tempo.

    Args:
        tp (int): original tempo

    Returns:
        int: normalized tempo
    """
    idx = np.argmin(np.abs(tp - tempo_bin))
    return int(tempo_bin[idx])


def norm_ts(ts, base=4):
    """Normalize time signature with denominator of 4.

    Args:
        ts (Fraction): _description_
        base (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    if isinstance(ts, str):
        ts = Fraction(ts)
    ts_num = ts.numerator * Fraction(base, ts.denominator)
    normed_ts = f"{ts_num}/{base}"
    return normed_ts


def norm_ts_tp(ts, tp):
    ratio = ts_tp_ratio(ts, tp)
    ts = Fraction(ts) * ratio
    normed_ts = norm_ts(ts)
    normed_tp = norm_tp(tp * ratio)
    return normed_ts, normed_tp


def ts_tp_ratio(ts, tp):
    ratio = 1
    ts_denom = int(ts.split("/")[-1])
    if tp <= 72 and ts_denom == 8:
        ratio = Fraction(2, 1)

    elif tp >= 192 and ts_denom == 2:
        ratio = Fraction(1, 2)

    return ratio


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


def remove_repeat(pattern):
    new_pattern = []
    last_sect = ''
    for sect in pattern:
        if sect != last_sect:
            new_pattern.append(sect)
        last_sect = sect
    return new_pattern


def load_event(fname):
    with open(fname) as f:
        event = json.load(f)

    note_event = {}
    for i in event['note']:
        note_event[int(i)] = event['note'][i].copy()

    struct = event['struct']

    return note_event, struct


def sort_section(norep_pattern):
    sorted_sects = []

    for sect in norep_pattern:
        if sect not in sorted_sects:
            sorted_sects.append(sect)
    return sorted_sects


def merge_pattern(pattern):
    merged_pattern = []
    merged_sect = []

    last_sect = pattern[0]
    for sect in pattern:
        if sect[0] == last_sect:
            merged_sect.append(sect)
        else:
            merged_pattern.append(merged_sect)
            merged_sect = [sect]
        last_sect = sect[0]

    if merged_sect:
        merged_pattern.append(merged_sect)

    return merged_pattern


def merge_section(sect_event, sects):
    merged_event = {}
    i_measure = 0

    for sect in sects:

        min_i_measure = min(sect_event[sect])
        max_i_measure = max(sect_event[sect])

        for i in range(min_i_measure, max_i_measure + 1):
            merged_event[i_measure] = sect_event[sect][i].copy()
            i_measure += 1

    return merged_event


def event_segment(event, start=(0, 0), end=(0, 0)):

    i_st, offset_st = start
    i_ed, offset_ed = end

    seg_event = {}
    for i_measure in range(i_st, i_ed):
        seg_event[i_measure] = event[i_measure].copy()

    # Add notes from last measure
    if offset_ed > 0:
        for i_token, token in enumerate(event[i_ed]['event']):
            if token[0] == 'o':
                if token2v(token) >= offset_ed:
                    break

        seg_event[i_ed] = event[i_ed].copy()
        seg_event[i_ed]['event'] = seg_event[i_ed]['event'][:i_token]

    # Remove redundant notes from the first measure
    if offset_st > 0:
        for i_token, token in enumerate(event[i_st]['event']):
            if token[0] == 'o':
                if token2v(token) >= offset_st:
                    break

        seg_event[i_st]['event'] = seg_event[i_st]['event'][i_token:]
    return seg_event


def phrase_segment(measures, max_len_phrase=8, measure_offset=0, phrase_offset=0):
    min_i_measure = min(measures) + measure_offset
    max_i_measure = max(measures)

    if measure_offset >= max_i_measure:
        return None

    measure = measures[min_i_measure]
    ts = measure['time_signature']
    tp = measure['tempo']

    normed_ts, normed_tp = norm_ts_tp(ts, tp)
    ts_token = f"ts-{normed_ts}"
    tp_token = f"tp-{normed_tp}"

    phrases = {}

    i_phrase = phrase_offset
    offset = token2v(measure['event'][0])
    len_phrase = 1 - offset / Fraction(ts) / 4

    phrases[i_phrase] = {"time_signature": ts_token,
                         "tempo": tp_token,
                         # "key": [measure["key"]],
                         "note": [measure['event']]}

    for i in range(min_i_measure + 1, max_i_measure + 1):

        notes = measures[i]['event']

        # Start a new phrase if time signature changes
        if measures[i]['time_signature'] != ts:

            # update time signature, tempo, offset
            ts = measures[i]['time_signature']
            tp = measures[i]['tempo']

            normed_ts, normed_tp = norm_ts_tp(ts, tp)
            ts_token = f"ts-{normed_ts}"
            tp_token = f"tp-{normed_tp}"

            offset = token2v(notes[0])

            # start a new phrase
            len_phrase = 1 - token2v(notes[0]) / Fraction(ts) / 4
            i_phrase += 1
            phrases[i_phrase] = {"time_signature": ts_token,
                                 "tempo": tp_token,
                                 "note": [notes],
                                 "key": [measures[i]["key"]]}

            continue

        # Phrase Boundary
        if len_phrase + 1 <= max_len_phrase:
            if i_phrase not in phrases:
                phrases[i_phrase] = {"time_signature": ts_token,
                                     "tempo": tp_token,
                                     "key": [],
                                     "note": []}

            phrases[i_phrase]['note'] += [notes]
            # phrases[i_phrase]['key'] += [measures[i]['key']]
            len_phrase += 1

        else:
            curr_notes = []
            # Add tokens to current phrase
            i_token = 0
            for i_token, token in enumerate(notes):

                if token[0] != 'o':
                    curr_notes.append(token)
                else:
                    onset = token2v(token)
                    if onset < offset:
                        curr_notes.append(f"o-{onset}")
                    else:
                        break

            phrases[i_phrase]['note'] += [curr_notes]
            # phrases[i_phrase]['key'] += [measures[i]['key']]

            # Start a new phrase with rest of the tokens in the current measure
            i_phrase += 1

            if i_token < len(notes) - 1:
                offset = token2v(notes[i_token])
                len_phrase = 1 - token2v(notes[0]) / Fraction(ts) / 4

                phrases[i_phrase] = {"time_signature": ts_token,
                                     "tempo": tp_token,
                                     #  "key": [measures[i]['key']],
                                     "note": [notes[i_token:]]}

            # No token left in current measure
            else:
                offset = 0
                len_phrase = 0

    return phrases


def get_events_per_section(event_file):
    # Load note event, and structure
    event, struct = load_event(event_file)

    # Remove repetition if the section has no first/second volta
    norep_pattern = remove_repeat(struct['pattern'])

    # Sort sections
    sects = sort_section(norep_pattern)

    # Get section onset for segmentation
    sect_onset = []
    for sect in sects:
        sect_onset.append(struct['attr'][sect])

    max_measure_id = max([i for i in event.keys()]) + 1
    sect_onset.append({"idx": max_measure_id, "onset": "o-0"})

    # Get events for each sub section
    sub_sect_event = {}
    for i, v in enumerate(sect_onset[:-1]):
        sect = sects[i]

        i_st = v['idx']
        offset_st = token2v(v['onset'])

        i_ed = sect_onset[i + 1]['idx']
        offset_ed = token2v(sect_onset[i + 1]['onset'])

        sub_sect_event[sect] = event_segment(event,
                                             start=(i_st, offset_st),
                                             end=(i_ed, offset_ed))

    # Merge events within sections such as [A, A1, A, A2]
    merged_pattern = merge_pattern(norep_pattern)

    # Get phrases for merged sections
    sect_event = {}
    for merged_sect in merged_pattern:
        sect_event[merged_sect[0]] = merge_section(sub_sect_event, merged_sect)

    return sect_event


def main(event_file, max_len_phrase=8, hop_size=4):

    sect_event = get_events_per_section(event_file)

    # Get phrases from section
    all_phrases = []
    for event in sect_event.values():

        # Key, time transpose
        event = normalize_event(event)

        for measure_offset in range(0, max_len_phrase, hop_size):
            phrases = phrase_segment(event, max_len_phrase, measure_offset)
            if not phrases:
                continue
            for phrase in phrases.values():
                all_phrases.append(phrase)

    return all_phrases


def mask_phrase(phrase, mask_mode='random', max_measure_len=64, mask_ratio=0.8,
                bar_eos_token='sep', eos_token='eos', pad_token='pad', pad_bar=True):

    tokens = [phrase['time_signature'], phrase['tempo']]

    n_measure = len(phrase['note'])

    # Randomly mask out measures
    if mask_mode == 'random':
        n_masked_meaure = int(mask_ratio * n_measure)
        masked_i_measures = random.sample(range(n_measure), n_masked_meaure)

    # Mask measures in the middle
    elif mask_mode == 'center':
        if n_measure < 3:
            return [], []
        else:
            # Mask around 50% bars in the middle of a sequence
            n_mask = int(n_measure / 2)
            mask_st_i = round(n_measure / 2 / 2)
            masked_i_measures = list(range(mask_st_i, mask_st_i + n_mask))

    masked_token_idx = []
    for i, measure in enumerate(phrase['note']):

        notes = measure
        if i in masked_i_measures:
            st_idx = len(tokens)

            if pad_bar:
                if len(notes) < max_measure_len:
                    pad_len = max_measure_len - len(notes) - 1
                    notes += [bar_eos_token] + \
                        [pad_token for _ in range(pad_len)]

            ed_idx = st_idx + len(notes)
            masked_token_idx += list(range(st_idx, ed_idx + 1))

        tokens += notes
        tokens += ['bar']

    if not pad_bar:
        tokens += [eos_token]
        
    return tokens, masked_token_idx


def make_masked_dataset(tokenizer, df, data_dir,
                        split='train', seq_len=512, mask_mode='random', mask_ratio=0.75, pad_bar=True):

    dataset = []

    for _, row in df[df['split'] == split].iterrows():
        fname = os.path.join(data_dir, row['fname'])

        with open(fname) as f:
            phrases = json.load(f)

        for phrase in phrases:

            tokens, mask_idx = mask_phrase(phrase,
                                           mask_mode=mask_mode,
                                           mask_ratio=mask_ratio,
                                           pad_bar=pad_bar,
                                           pad_token=tokenizer.pad_token,
                                           bar_eos_token=tokenizer.sep_token,
                                           eos_token=tokenizer.eos_token)

            if not len(mask_idx):
                continue

            if has_irregular_token(tokens, tokenizer.vocab):
                continue

            if len(tokens) >= seq_len:  # roberta pos id starts from 1
                continue

            ids = np.array(tokenizer.convert_tokens_to_ids(tokens))

            input_ids = ids.copy()
            input_ids[mask_idx] = tokenizer.mask_id

            labels = np.array([tokenizer.mask_id for _ in range(len(ids))])
            labels[mask_idx] = ids[mask_idx]

            entry = {"input_ids": input_ids.tolist(),
                     "labels": labels.tolist()}

            dataset.append(entry)

    return dataset
