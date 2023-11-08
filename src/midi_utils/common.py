"""Helper functions for loading data and evaluation.
"""
import pretty_midi
from copy import deepcopy


def pretty_midi_sort(pm):
    """
    Sort notes/control changes by time in place
    """
    for i in range(len(pm.instruments)):
        pm.instruments[i].notes = sorted(
            pm.instruments[i].notes, key=lambda note: note.start)
        pm.instruments[i].control_changes = sorted(
            pm.instruments[i].control_changes, key=lambda event: event.time)
    return


def trim_midi(pm, t_start, t_end, is_sorted=True, meta=True, cut_by='offset'):
    """Trim midi given the start and end time.

    Args:
        midi (PrettyMIDI): pretty_midi loaded by pretty midi. Assume that notes are sorted by start time.
        t_start (float): starting time in second.
        t_end (float): ending time in second.
        is_sorted (Optional, bool): whether notes have been sorted. 

    Returns:
        PrettyMIDI: Sliced pretty_midi.
    """
    if not is_sorted:
        pretty_midi_sort(pm)

    # Initial tempo
    prev_tempo = [tempo for t, tempo in zip(*pm.get_tempo_changes())
                  if t <= t_start]
    initial_tempo = prev_tempo[-1]

    # Initialize pm
    pm_slice = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo,
                                      resolution=pm.resolution)

    if meta:
        # Todo: fix transfer tick scales
        start_tick = pm.time_to_tick(t_start)
        end_tick = pm.time_to_tick(t_end)

        n_change = len(pm._tick_scales)

        p = 0  # points to the last tempo change before start_tick
        while p < n_change and pm._tick_scales[p][0] < start_tick:
            p += 1
        q = p  # points to the last tempo change before end_tick
        while q < n_change and pm._tick_scales[q][0] < end_tick:
            q += 1

        tick_scales = pm._tick_scales[max(0, p - 1):q]
        pm_slice._tick_scales = list(map(lambda x: (max(0, int(x[0] - start_tick)), x[1]),
                                         tick_scales))

        # Initialize meta data
        # Todo: Modulize this
        # Time Signature
        prev_ts_obj = None
        for ts_obj in pm.time_signature_changes:
            if ts_obj.time < t_start:
                prev_ts_obj = pretty_midi.TimeSignature(numerator=ts_obj.numerator,
                                                        denominator=ts_obj.denominator,
                                                        time=0)
                continue
            if ts_obj.time > t_end:
                break
            new_ts_obj = pretty_midi.TimeSignature(numerator=ts_obj.numerator,
                                                   denominator=ts_obj.denominator,
                                                   time=ts_obj.time - t_start)
            pm_slice.time_signature_changes.append(new_ts_obj)
        if prev_ts_obj is not None:
            pm_slice.time_signature_changes.append(prev_ts_obj)

        prev_key_obj = None
        for key_obj in pm.key_signature_changes:
            if key_obj.time < t_start:
                prev_key_obj = pretty_midi.KeySignature(key_number=key_obj.key_number,
                                                        time=0)
                continue
            if key_obj.time > t_end:
                break
            new_key_obj = pretty_midi.KeySignature(key_number=key_obj.key_number,
                                                   time=key_obj.time - t_start)
            pm_slice.key_signature_changes.append(new_key_obj)

        if prev_key_obj is not None:
            pm_slice.key_signature_changes.append(prev_key_obj)

    # Note start end time in original pm
    prev_t_st = None

    # Looping through all instruments

    for orig_inst in pm.instruments:
        inst = pretty_midi.Instrument(program=orig_inst.program,
                                      is_drum=orig_inst.is_drum,
                                      name=orig_inst.name)
        for note in orig_inst.notes:
            if note.start < t_start or note.end < t_start:
                continue
            if note.start > t_end:
                break

            if cut_by == 'onset':
                note_end = note.end - t_start
            elif cut_by == 'offset':
                note_end = min(t_end - t_start, note.end - t_start)

            # if prev_t_st is None:
            #     prev_t_st = note.start
            # elif note.start < prev_t_st:
            #     prev_t_st = note.start

            new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch,
                                        start=max(0, note.start - t_start),
                                        end=note_end)
            inst.notes.append(new_note)

        for ctrl in orig_inst.control_changes:
            if ctrl.time >= t_start and ctrl.time < t_end:
                new_ctrl = pretty_midi.ControlChange(number=ctrl.number,
                                                     value=ctrl.value,
                                                     time=ctrl.time - t_start)
                inst.control_changes.append(new_ctrl)

        pm_slice.instruments.append(inst)
    return pm_slice, prev_t_st


def strip_midi(pm, is_sorted=True):
    if not is_sorted:
        pretty_midi_sort(pm)

    t_offset = pm.get_end_time()
    for inst in pm.instruments:
        t_offset = min(inst.notes[0].start, t_offset)

    for inst in pm.instruments:
        for note in inst.notes:
            note.start -= t_offset
            note.end -= t_offset

    return pm, t_offset


def change_pitch(pm, pitch_shift, inplace=True):
    """Pitch transpose by `pitch_shift` in chromatic scale.

    Args:
        pm (PrettyMIDI): Original PrettyMIDI object.
        pitch_shift (int): Pitch shift in in chromatic scale.
        inplace (bool, optional): Perform pitch transpose in place or make a copy of the Original. Defaults to True.

    Returns:
        The pitch transposed PrettyMIDI object.
    """
    pitch_shift = int(pitch_shift)

    if not inplace:
        new_pm = deepcopy(pm)
    else:
        new_pm = pm

    for inst in new_pm.instruments:
        for note in inst.notes:
            note.pitch += pitch_shift
    return new_pm


def change_tempo(pm, ratio=1.0):
    """Transpose the tempo of a given prettyMIDI object.

    Args:
        pm (pretty_midi.PrettyMIDI): original PrettyMIDI object
        ratio (float, optional): ratio = (new tempo)/(original tempo). Defaults to 1.0.

    Returns:
        A new PrettyMIDI object
    """
    new_pm = pretty_midi.PrettyMIDI()

    for orig_inst in pm.instruments:
        inst = pretty_midi.Instrument(program=orig_inst.program,
                                      is_drum=orig_inst.is_drum,
                                      name=orig_inst.name)

        for orig_note in orig_inst.notes:
            note = pretty_midi.Note(velocity=orig_note.velocity,
                                    pitch=orig_note.pitch,
                                    start=orig_note.start / ratio,
                                    end=orig_note.end / ratio)
            inst.notes.append(note)

        new_pm.instruments.append(inst)

    return new_pm
