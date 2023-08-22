import json
import math
import numpy as np
from fractions import Fraction


def normalize_tp(tp, base=24):
    return math.ceil(tp / base) * base


def normalize_ts(ts_num, ts_denom, base=4):
    new_ts_num = Fraction(ts_num) * Fraction(base, ts_denom)
    norm_ts = f"{new_ts_num}/{base}"
    return norm_ts


def get_onset(event, prefix='o'):
    return Fraction(event.split(f'{prefix}-')[-1])


def normalize_event(events, denom=4, num=4):
    ratio = Fraction(4, denom)
    norm_events = []
    for event in events:
        if event[0] == 'o':
            # normalize to 1
            # onset = get_onset(event) / Fraction(num)

            # normalize to quarter note
            onset = get_onset(event) * ratio
            norm_events.append(f"o-{onset}")
        else:
            norm_events.append(event)

    return norm_events


def get_phrases(measures, n_bar_per_phrase=8):

    n_measure = len(measures)

    # first_bar
    measure = measures['0']

    # initial tempo, time signature
    tp = measure['tempo']
    ts_num = measure['time_signature']['numerator']
    ts_denom = measure['time_signature']['denominator']
    ts = f'{ts_num}/{ts_denom}'

    event = measure['event']
    offset = get_onset(event[0]) / ts_num

    n_bar = offset
    curr_phrase = event + ['bar']
    # curr_phrase = normalize_event(event, ts_denom) + ['bar']

    phrase_id = 0
    phrases = {}

    for measure_idx in range(1, n_measure):

        measure = measures[str(measure_idx)]
        event = measure['event']

        if not len(event):
            phrases[phrase_id] = [f'ts-{norm_ts}',
                                  f'tp-{norm_tp}'] + curr_phrase
            phrase_id += 1
            curr_phrase = []
            continue

        # or meter changes
        new_ts_num = measure['time_signature']['numerator']
        new_ts_denom = measure['time_signature']['denominator']
        new_ts = f'{new_ts_num}/{new_ts_denom}'

        # Start a new phrase if time signature changes
        if new_ts != ts or measure['tempo'] != tp:

            # Add the previous phrase
            norm_ts = normalize_ts(ts_num, ts_denom)
            norm_tp = normalize_tp(tp)
            phrases[phrase_id] = [f'ts-{norm_ts}',
                                  f'tp-{norm_tp}'] + curr_phrase
            phrase_id += 1

            # update tempo and time signature
            tp = measure['tempo']
            ts_num, ts_denom = new_ts_num, new_ts_denom
            ts = new_ts

            # start a new phrase
            # curr_phrase = normalize_event(event, ts_denom) + ['bar']
            # curr_phrase = normalize_event(event, num=ts_num) # norm to 1
            curr_phrase = event + ['bar']

            # update bar offset
            offset = get_onset(event[0]) / ts_num
            continue

        if n_bar + 1 >= n_bar_per_phrase:

            # Add tokens to previous phrase
            for token_id, token in enumerate(event):
                if token[0] != 'o':
                    curr_phrase.append(token)
                else:
                    onset = get_onset(token) / ts_num
                    if onset < offset:
                        curr_phrase.append(f"o-{onset}")
                    else:
                        break

            norm_ts = normalize_ts(ts_num, ts_denom)
            norm_tp = normalize_tp(tp)
            phrases[phrase_id] = [f'ts-{norm_ts}',
                                  f'tp-{norm_tp}'] + curr_phrase
            phrase_id += 1

            # Start a new phrase with rest of the tokens in the current measure
            n_bar = offset

            if token_id < len(event) - 1:
                # curr_phrase = normalize_event(event[token_id:], ts_num)
                # curr_phrase = normalize_event(event[token_id:], ts_denom)
                # curr_phrase += ['bar']
                curr_phrase = event[token_id:] + ['bar']

            # If no token left
            else:
                curr_phrase = []

        else:
            # curr_phrase += normalize_event(event, ts_num)
            # curr_phrase += normalize_event(event, ts_denom)
            # curr_phrase += ['bar']
            curr_phrase += event + ['bar']
            n_bar += 1

    if len(curr_phrase):
        norm_ts = normalize_ts(ts_num, ts_denom)
        norm_tp = normalize_tp(tp)
        phrases[phrase_id] = [f'ts-{norm_ts}', f'tp-{norm_tp}'] + curr_phrase

    return phrases


def validate_phrase(phrase):
    i = 0
    while i < len(phrase):
        if phrase[i][0] == 'o':
            break
        else:
            i += 1

    j = len(phrase)
    while j > 0:
        if phrase[j - 1][0] == 'd':
            break
        elif phrase[j - 1] == 'bar':
            break
        elif phrase[j - 1] == 'eos':
            break
        else:
            j -= 1
    return phrase[i:j]


def split_phrase(phrase, seq_len=256):
    ts_tp, tokens = phrase[:2], phrase[2:]
    bar_pos = np.where(np.array(tokens) == 'bar')[0]

    phrases = []
    st = 0
    for i in range(1, len(bar_pos)):
        if bar_pos[i] - st + 3 > seq_len:
            segment = validate_phrase(tokens[st: bar_pos[i - 1] + 1])
            if len(segment) > seq_len - 1:
                segment = validate_phrase(segment[-seq_len + 2:])
            phrases.append(ts_tp + segment)
            st = bar_pos[max(0, i - 2)]

    if len(tokens) - st + 2 > seq_len:
        segment = ts_tp + validate_phrase(tokens[-seq_len + 3:])
    else:
        segment = ts_tp + validate_phrase(tokens[st:])

    phrases.append(segment)
    return phrases

# fname = ''
# with open(fname) as f:
#     measures = json.load(f)
