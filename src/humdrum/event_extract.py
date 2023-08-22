"""Extract notes (grouped by measures), tempo, time signature from humdrum file, and output a dictionary that contains following attributes:
{
    "note": {
        `measure_id`:{
            "event": [`note onset`, `pitch`, `note duration`],
            "tempo": `tempo`,
            "time_signature": `time signature`,
            "key": `key`
            },
    "struct": {
        "pattern": `structure pattern`,
        "attr": {
            `section`: {"id": `measure index of section onset`,
                        "start_pos": `start position within a measure`}
            }
        }
}

The note onset and duration are scaled by quarter note.

Humdrum file could be loaded
(1) directly as .krn file
(2) as .xml converted by `hum2xml`.
Method (2) is recommended because music21 is not very reliable handling .krn file.

"""
import re
import math
import music21
from fractions import Fraction
from music21 import key, stream, pitch
# environment.set("musescoreDirectPNGPath", "/Applications/MuseScore 4.app/Contents/MacOS/mscore")


def is_kern_note(entry):
    """Check if a kern entry is a note, i.e. follows the pattern such as `4C`, `4.C`.

    Args:
        entry (str): _description_

    Returns:
        bool: _description_
    """
    is_note = False

    entry = entry.replace("(", "")
    entry = entry.replace("[", "")
    entry = entry.replace("<", "")

    is_full = bool(re.match(r'[0-9]+[a-zA-Z]', entry))
    is_dot = bool(re.match(r'[0-9]]+[.][a-zA-Z]', entry))

    if is_full or is_dot:
        is_note = True

    return is_note


def entries_has_note(entries):
    """For sanity check. Check if there is note within given entries.

    Args:
        entries (list): _description_

    Returns:
        bool: _description_
    """

    has_note = False
    for entry in entries:
        has_note = any([is_kern_note(item) for item in entry.split("\t")])

        if has_note:
            break

    return has_note


def norm_pitch(pitch_str):
    """Make sure equivalent pitches have identical annotation.

    Args:
        pitch_str (str)

    Returns:
        str: normalized annotation
    """
    note_ps = pitch.Pitch(pitch_str).ps
    return pitch.Pitch(note_ps).nameWithOctave


def flatten_event(note_event):
    """Flatten a list of note events into a list of tokens. Events with same onset share the same onset tokens.

    Example 1
    Input: [[onset_1, pitch_1, duration_1], [onset_2, pitch_2, duration_2]]
    Output: [onset_1, pitch_1, duration_1, onset_2, pitch_2, duration_2]

    Example 2
    Input: [[onset_1, pitch_1, duration_1], [onset_1, pitch_2, duration_2]]
    Output: [onset_1, pitch_1, duration_1, pitch_2, duration_2]

    Args:
        note_event (list): _description_

    Returns:
        list: _description_
    """
    if not len(note_event):
        return []

    note_event = sorted(note_event, key=lambda x: (x[0], x[1], -x[2]))
    last_onset = Fraction(-1)
    flatten_event = []

    for entry in note_event:

        onset_token = f"o-{entry[0]}"
        pitch_token = entry[1]
        dur_token = f"d-{entry[2]}"

        if entry[0] != last_onset:
            flatten_event.append(onset_token)

            flatten_event += [pitch_token, dur_token]
            last_onset = entry[0]
        elif flatten_event[-2] != pitch_token:

            flatten_event += [pitch_token, dur_token]

    return flatten_event


def get_key(measure, mode="major"):
    """Get key string from music21.measure. 

    Args:
        measure (_type_): _description_

    Returns:
        str: Key signature
    """
    ks = None

    for i in measure.getElementsByClass(key.Key).elements:
        ks = i.name

    # If key is not directly available, convert key from key signature.
    if not ks:
        ks_str = measure.keySignature
        if ks_str:
            ks = ks_str.asKey(mode=mode).name

    return ks


def check_tempo_shift(event, attr, measure_offset=0):
    """Check if tempo extracted from .xml is consistent with .krn

    Args:
        event (dict): _description_
        attr (dict): _description_
    """

    max_xml_measure = max(event.keys())
    min_xml_measure = min(event.keys())

    ts_shift = sorted(attr['time_signature'].keys())
    ts_shift.append(max_xml_measure + measure_offset + 1)
    ts_pos = 0

    for i in range(min_xml_measure, max_xml_measure + 1):
        if i + measure_offset >= ts_shift[ts_pos + 1]:
            ts_pos += 1
        event_ts = event[i]['time_signature']
        krn_ts = attr['time_signature'][ts_shift[ts_pos]]
        assert event_ts == krn_ts, f"{i}"

    return


def get_ts_tp(krn_entry, init_i_measure=1):
    """Get time signature and tempo from humdrum entry.

    Args:
        krn_entry (list): _description_
        init_i_measure (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    ts_measure_idx = {}
    tp_measure_idx = {}

    i_measure = init_i_measure

    for entry in krn_entry:

        entry = entry.split("\t")[0]

        match_measure = re.match(r"=[0-9]+", entry)
        if bool(match_measure):
            i_measure = int(entry[1: match_measure.end()])

        if entry[:3] == "*MM":
            tp_measure_idx[i_measure] = int(entry[3:])

        elif entry[:2] == "*M":
            ts_measure_idx[i_measure] = entry[2:]

        else:
            continue
    return ts_measure_idx, tp_measure_idx


def get_struct_pattern(krn_entry):
    """Get structure pattern, with repeation from humdrum entry.

    Args:
        krn_entry (_type_): _description_

    Returns:
        list: an array of sections
    """

    pattern = []

    for entry in krn_entry:
        entry = entry.split("\t")[0]
        if entry[:3] == "*>[":
            assert entry[-1] == "]"
            sect_str = entry.split("\t")[0]
            pattern = sect_str[3:-1].split(",")
            break

    return pattern


def get_sect_measure_idx(krn_entry):
    """Get the onset (measure id) of each section.

    Args:
        krn_entry (list): _description_

    Returns:
        dict: `idx`: onset measure id
              `start_from_0`: whether the section starts from the downbeat of a measure.
    """
    sect_measure_idx = {}

    for i, entry in enumerate(krn_entry):

        if entry[:2] == "*>":
            sect_str = entry.split("\t")[0].replace("*>", "")

            j = i - 1

            last_entry = krn_entry[j].split("\t")[0]
            matched = re.match(r"=[0-9]+", last_entry)

            while not bool(matched):
                j -= 1
                last_entry = krn_entry[j].split("\t")[0]
                matched = re.match(r"=[0-9]+", last_entry)

            if j != i - 1:
                has_note = entries_has_note(krn_entry[j:i])
            else:
                has_note = False

            # start_from_0 = (j == i - 1)
            i_measure = int(last_entry[1: matched.end()])

            sect_measure_idx[sect_str] = {"idx": i_measure,
                                          "start_from_0": not has_note}

    return sect_measure_idx


def krn_attr_extract(krn_file):
    """Read attributes such pattern, tempo, time signature from .krn file.

    NOTE: 
    * tempo is read from .krn directly because it is no available in the .xml converted from .krn using `hum2xml`
    * time signature is read for sanity check

    Args:
        krn_file (str): file name 

    Returns:
        dict: _description_
    """

    with open(krn_file) as f:
        krn_entry = f.read().splitlines()

    # Get init measure index
    init_measure_str = None
    for i, entry in enumerate(krn_entry):

        if entry[0] == "=":
            measure_start_idx = i
            init_measure_str = entry.split("\t")[0]
            break

    has_note = entries_has_note(krn_entry[: measure_start_idx])

    matched = re.match(r'=[0-9]+', init_measure_str)

    if bool(matched):
        init_i_measure = int(init_measure_str[1: matched.end()])
        if has_note:
            init_i_measure -= 1
        # matched = re.match(r'=[0-9]+[-]', init_measure_str)
        # if not bool(matched):
        #     init_i_measure -= 1
    else:
        init_i_measure = 0

    ts, tp = get_ts_tp(krn_entry, init_i_measure)

    # Get structure pattern
    pattern = get_struct_pattern(krn_entry)
    if not len(pattern):
        # Warning(f"No structure pattern found in {krn_file}")
        pattern = ["A"]

    krn_entry = krn_entry[measure_start_idx + 1:]

    # Get measure index corresponds of section onset
    if len(pattern) > 1:
        sect_measure_idx = get_sect_measure_idx(krn_entry)
    else:
        sect_measure_idx = {}

    first_sect = pattern[0]
    if first_sect not in sect_measure_idx:
        sect_measure_idx[first_sect] = {"idx": init_i_measure,
                                        "start_from_0": not has_note}

    krn_attr = {"pattern": pattern,
                "attr": sect_measure_idx,
                "time_signature": ts,
                "tempo": tp}

    return krn_attr


def part_event_extract(part):
    """Extract notes grouped by measures from music21.part, with quarter note as time unit.
    Assume that measure index starts from either 0  of 1 (start from downbeat).

    Args:
        part (music21.part): _description_

    Raises:
        ValueError: If not time signature is found.

    Returns:
        dict: {"measure_id": [[note offset, pitch, duration]]}
    """

    # NOTE: such extension should work as long as the key is not played by both hands
    event = {}
    to_extend = {}

    measures = part.getElementsByClass(stream.Measure)

    # Initial Time Signature
    ts = measures[0].timeSignature
    if not ts:
        raise ValueError("No Time Signature")

    # measure duration in quarter length
    bar_dur = Fraction(ts.barDuration.quarterLength)
    note_offset = Fraction(measures[0].offset)
    pos_offset = Fraction(measures[0].quarterLength) - bar_dur

    if pos_offset != 0:
        bar_offset = 0
    else:
        bar_offset = 1

    last_bar = 0

    # Extract events from each measure/bar
    # NOTE: Assume that the measures are ordered by offset
    for i, measure in enumerate(measures):

        # time signature
        if measure.timeSignature and measure.timeSignature.ratioString != ts.ratioString:
            ts = measure.timeSignature
            bar_dur = Fraction(ts.barDuration.quarterLength)

            # NOTE: measure.number is not reliable
            # If time signature changes, starting a new bar anyway
            bar = last_bar + 1
            note_offset = Fraction(measure.offset)
            pos_offset = Fraction(measure.quarterLength) - bar_dur
            bar_offset = bar
        else:
            bar = math.floor((measure.offset - note_offset -
                             pos_offset) / bar_dur) + bar_offset

        # Update key signature
        curr_ks = get_key(measure)
        ks = curr_ks or ks

        last_bar = bar

        if bar not in event:
            event[bar] = {"event": [],
                          "key": ks,
                          "duration": [Fraction(measure.quarterLength)],
                          "time_signature": ts.ratioString}
        else:
            event[bar]["duration"] += [Fraction(measure.quarterLength)]

        for note in measure.flatten().notes:

            if note.duration.isGrace:
                continue

            quart_note = Fraction(note.offset) + \
                Fraction(measure.offset) - note_offset
            offset = (quart_note - pos_offset) % bar_dur

            if note.isChord:

                for chord_note in note.notes:
                    pitch_name = norm_pitch(chord_note.pitch.nameWithOctave)
                    note_duration = Fraction(chord_note.duration.quarterLength)

                    if not chord_note.tie:
                        event[bar]["event"].append(
                            [offset, pitch_name, note_duration])

                    elif chord_note.tie.type == "start":
                        to_extend[pitch_name] = [bar, offset, note_duration]

                    elif chord_note.tie.type == "continue":
                        if pitch_name in to_extend:
                            to_extend[pitch_name][-1] += note_duration
                        else:
                            to_extend[pitch_name] = [
                                bar, offset, note_duration]

                    elif chord_note.tie.type == "stop":

                        if pitch_name in to_extend:
                            to_extend[pitch_name][-1] += note_duration

                            # Clean up to_extend and update events
                            prev_bar = to_extend[pitch_name][0]
                            # bar, offset, duration
                            entry = to_extend.pop(pitch_name)
                            event[prev_bar]["event"].append(
                                [entry[1], pitch_name, entry[2]])
                        else:
                            event[bar]["event"].append(
                                [offset, pitch_name, note_duration])

            else:
                pitch_name = norm_pitch(note.pitch.nameWithOctave)
                note_duration = Fraction(note.duration.quarterLength)

                if not note.tie:
                    event[bar]["event"].append(
                        [offset, pitch_name, note_duration])

                elif note.tie.type == "start":
                    to_extend[pitch_name] = [bar, offset, note_duration]

                elif note.tie.type == "continue":
                    if pitch_name in to_extend:
                        to_extend[pitch_name][-1] += note_duration
                    else:
                        to_extend[pitch_name] = [bar, offset, note_duration]

                elif note.tie.type == "stop":
                    if pitch_name in to_extend:
                        to_extend[pitch_name][-1] += note_duration

                        # Clean up to_extend and update event
                        prev_bar = to_extend[pitch_name][0]
                        # bar, offset, duration
                        entry = to_extend.pop(pitch_name)
                        event[prev_bar]["event"].append(
                            [entry[1], pitch_name, entry[2]])
                    else:
                        # bar 41, Sonata No. 7 in C major, K 309 / K 284b, 2.
                        event[bar]["event"].append(
                            [offset, pitch_name, note_duration])
                        continue

    return event


def event_extract(krn_file, mxml_file=None):

    # Load music scores from .xml or .krn
    if not mxml_file:
        mxml_file = krn_file
    s = music21.converter.parse(mxml_file)
    parts = s.getElementsByClass(music21.stream.Part)

    # Extract structure pattern and attribute
    krn_attr = krn_attr_extract(krn_file)
    init_sect = krn_attr["pattern"][0]
    krn_measure_offset = krn_attr["attr"][init_sect]["idx"]

    # Extract notes
    event = {}
    measure_offset = 0
    for part in parts:
        event_part = part_event_extract(part)
        measure_offset = krn_measure_offset - min(event_part.keys())

        # Sanity check
        check_tempo_shift(event_part, krn_attr, measure_offset)

        # Update events
        if not len(event):
            event = event_part.copy()
            continue

        for i, attr in event_part.items():

            if i not in event:
                Warning(f"Create measure #{i} in {mxml_file}")
                event[i] = {"event": [],
                            "key": None,
                            "tempo": None,
                            "time_signature": None,
                            "duration": []}

            event[i]["event"] = event[i]["event"] + attr["event"]
            event[i]["key"] = event[i]["key"] or attr["key"]
            event[i]["time_signature"] = event[i]["time_signature"] or attr["time_signature"]
            event[i]["duration"] = event[i]["duration"] or attr["duration"]

    sorted_event = {}
    sorted_event["note"] = {}

    tp_shift = sorted(krn_attr["tempo"])
    tp_shift.append(max(event.keys()) + 1)
    tp_pos = 0

    for i in sorted(event.keys()):

        # Sort notes and flatten event entries
        sorted_event["note"][i] = {}
        sorted_event["note"][i]["event"] = flatten_event(event[i]["event"])

        # Add time signature
        sorted_event["note"][i]["time_signature"] = event[i]["time_signature"]

        # Add tempo
        if i >= tp_shift[tp_pos + 1]:
            tp_pos += 1
        sorted_event["note"][i]["tempo"] = krn_attr["tempo"][tp_shift[tp_pos]]

        # Add key
        sorted_event["note"][i]["key"] = event[i]["key"]

    sorted_event["struct"] = {}
    sorted_event["struct"]["pattern"] = krn_attr["pattern"]
    sorted_event["struct"]["attr"] = {}

    for sect, pos in krn_attr["attr"].items():

        i = pos["idx"] - measure_offset
        measure_ts = event[i]["time_signature"]
        measure_dur = Fraction(measure_ts) * 4
        measure_split = event[i]["duration"]

        if not pos["start_from_0"]:
            start_pos = measure_dur - measure_split[-1]
            assert start_pos != measure_dur, f"no a bar line found in measure {i}"
            assert start_pos != 0, f"section {sect} in measure {i} should not start from 0"
        else:
            start_pos = Fraction(0)

        sorted_event["struct"]["attr"][sect] = {"idx": i,
                                                "onset": f"o-{start_pos}"}

    return sorted_event


if __name__ == "__main__":

    krn_file = "../../sonata-dataset/krn/beethoven/sonata13-1.krn"
    # import os
    # import json
    # import pandas as pd

    # composers = ["mozart"]
    # data_dir = "../data/sonata-dataset"
    # output_dir = os.path.join(data_dir, "events")
    # os.makedirs(output_dir, exist_ok=True)

    # for composer in composers:
    #     df = pd.read_csv(os.path.join(data_dir, f"{composer}.csv"))
    #     os.makedirs(os.path.join(output_dir, composer), exist_ok=True)

    #     for _, row in df.iterrows():
    #         prefix = row["filename"]
    #         krn_file = os.path.join(data_dir, "krn", composer, f"{prefix}.krn")
    #         event_file = os.path.join(output_dir, composer, f"{prefix}.json")

    #         try:
    #             event = krn_event_extract(krn_file)
    #             with open(event_file, "w") as f:
    #                 json.dump(event, f)
    #         except:
    #             print(f"Event extraction failed: {krn_file}.")
