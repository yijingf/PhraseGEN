"""
Phrase annotation and segmentation based on McFee's implementation of the laplacian segmentation method in `McFee and Ellis, 2014 <http://bmcfee.github.io/papers/ismir2014_spectral.pdf>`_,
with a couple of minor stability improvements.

The phrase boundaries calculated on audio are later post-processed by
    1. merging phrases that are too short 
    2. aligning with the corresponding midi
    3. obtaining phrase relations for next phrase prediction tasks

Outputs:
    1. DATA_DIR/raw_phrase_boundaries.json: Raw phrase boundaries;
    2. DATA_DIR/phrase_boundaries.json: Post process phrase boundaries;
    3. DATA_DIR/phrase_boundaries_failed.json: indices of failed pieces;
    4. DATA_DIR/next_phrase_index.json: Indices of phrases and next-phrase relations;
    5. DATA_DIR/midi_phrase/*.mid: MIDI phrases if `save_midi == True`.

Usage: python3 audio_segment.py

"""
# Modified: Yijing Feng
# Code source: Brian McFee
# License: ISC

import os
import scipy
import librosa
import pretty_midi
import numpy as np
import pandas as pd
import sklearn.cluster

import sys
sys.path.append("../..")
from midi_utils.constants import MAESTRO_DIR, DATA_DIR
from midi_utils.common import pretty_midi_sort, trim_midi, strip_midi

AUDIO_DIR = os.path.join(MAESTRO_DIR, "audio")
MIDI_DIR = os.path.join(MAESTRO_DIR, "midi")
MIDI_PHRASE_DIR = os.path.join(DATA_DIR, "midi_phrase")

BINS_PER_OCTAVE = 12 * 3
N_OCTAVES = 7


def laplacian_segment(y, sr=22050, k=5):
    # Compute the log-power CQT
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr,
                                                   bins_per_octave=BINS_PER_OCTAVE,
                                                   n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
                                ref=np.max)

    # To reduce dimensionality, we'll beat-synchronous the CQT
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # For plotting purposes, we'll need the timing of the beats
    # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
    beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
                                                                x_min=0),
                                        sr=sr)

    # Let's build a weighted recurrence matrix using beat-synchronous CQT
    # (Equation 1)
    # width=3 prevents links within the same bar
    # mode='affinity' here implements S_rep (after Eq. 8)
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                          sym=True)

    # Enhance diagonals with a median filter (Equation 2)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    # Now let's build the sequence matrix (S_loc) using mfcc-similarity
    #
    #   :math:`R_\text{path}[i, i\pm 1] = \exp(-\|C_i - C_{i\pm 1}\|^2 / \sigma^2)`
    #
    # Here, we take :math:`\sigma` to be the median distance between successive beats.
    #
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)

    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)

    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # And compute the balanced combination (Equations 6, 7, 9)
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    A = mu * Rf + (1 - mu) * R_path

    # Now let's compute the normalized Laplacian (Eq. 10)
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)

    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

    # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5

    # If we want k clusters, use the first k normalized eigenvectors.
    # Fun exercise: see how the segmentation changes as you vary k
    # Basically: Lager K, finer segments
    X = evecs[:, :k] / Cnorm[:, k - 1:k]

    # Let's use these k components to cluster beats into segments
    # (Algorithm 1)
    KM = sklearn.cluster.KMeans(n_clusters=k)

    seg_ids = KM.fit_predict(X)
    seg_ids = seg_ids[:len(beats)]  # sync might compute one more beat

    # Locate segment boundaries from the label sequence
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beat 0 as a boundary
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    # bound_segs = list(seg_ids[bound_beats])

    # Convert beat indices to frames
    bound_frames = beats[bound_beats]

    # Make sure we cover to the end of the track
    bound_frames = librosa.util.fix_frames(bound_frames,
                                           x_min=None,
                                           x_max=C.shape[1] - 1)

    bound_times = librosa.frames_to_time(bound_frames)

    return bound_times


def merge_phrase(t_bound, t_thresh=4):
    """Merge phrases shorter than t_thresh

    Args:
        t_bound (list): A list of phrase boundaries.
        t_thresh (int, optional): Phrase duration threshold. Defaults to 4.

    Returns:
        Merged phrase boundaries.
    """

    last_t = t_bound[0]
    merged_t_bound = [last_t]

    for t in t_bound[1:]:
        if t - last_t > t_thresh:
            merged_t_bound.append(t)
            last_t = t

    if t != last_t:
        merged_t_bound.append(t)

    return merged_t_bound


def next_phrase(pm, t_bound, max_t_phrase=32, min_t_phrase=4, t_phrase_interval=4,
                save_midi=True, midi_prefix=''):
    """Get phrase relation for next-phrase prediction tasks.

    Args:
        pm (pretty_midi.PrettyMIDI): Input
        t_bound (list): A list of phrase boundaries.
        max_t_phrase (float, optional): Max phrase length in seconds. Defaults to 32.
        min_t_phrase (float, optional): Min phrase length in seconds. Defaults to 4.
        t_phrase_interval (float, optional): Max interval between phrases in seconds. Defaults to 4.
        save_midi (bool, optional): Write midi phrase to file. Defaults to True.
        midi_prefix (str, optional): MIDI phrase file name prefix. Defaults to ''.

    Returns:
        phrase index, pairs of Next-phrases indices
    """

    pair = []
    phrase_idx = []
    pairs_idx = []

    pretty_midi_sort(pm)

    for i, (t_st, t_ed) in enumerate(zip(*(t_bound[:-1], t_bound[1:]))):

        # Skip extremely long/short phrase
        t_seg = t_ed - t_st
        if t_seg > max_t_phrase or t_seg < min_t_phrase:
            continue

        # Trim midi
        pm_seg, _ = trim_midi(pm, t_st, t_ed, cut_by='onset', meta=False)

        # Skip empty phrase
        if not len(pm_seg.instruments[0].notes):
            continue

        pm_seg, strip_t = strip_midi(pm_seg)

        # Skip short phrase again
        if pm_seg.get_end_time() < min_t_phrase:
            continue

        # Whether the current phrase is a continuation of last phrase
        if strip_t < t_phrase_interval and len(pair):
            if i - pair[-1] == 1:
                pairs_idx.append(pair + [i])

        pair = [i]

        # If the current phrase is followed by `min_t_phrase` silence, then the next phrase is not likely to be it's continuation.
        if t_seg - strip_t - pm_seg.get_end_time() > min_t_phrase:
            pair = []

        phrase_idx.append(i)

        if save_midi:
            pm_seg.write(os.path.join(
                MIDI_PHRASE_DIR, f"{midi_prefix}-{i}.mid"))

    return phrase_idx, pairs_idx


def main(save_midi=True):
    info_df = pd.read_csv(os.path.join(MAESTRO_DIR, "maestro-v3.0.0.csv"))

    raw_phrase_boundary = {}
    phrase_boundary = {}
    failed = {}
    phrase_idx_dict = {}

    for i, row in info_df.iterrows():
        # Load Audio
        audio_file = os.path.join(AUDIO_DIR, row['audio_filename'])
        y, sr = librosa.load(audio_file)

        # Load single track MIDI
        midi_file = os.path.join(MIDI_DIR, row['midi_filename'])
        pm = pretty_midi.PrettyMIDI(midi_file)

        try:
            raw_t_bound = laplacian_segment(y).tolist()

            # Align with midi
            midi_t_end = pm.get_end_time()
            if raw_t_bound[-1] >= midi_t_end:
                t_bound = [0] + raw_t_bound[:-1] + [midi_t_end]
            else:
                t_bound = [0] + raw_t_bound + [midi_t_end]

            raw_phrase_boundary[i] = raw_t_bound

            # Merge short phrases
            t_bound = merge_phrase(t_bound)
            phrase_boundary[i] = t_bound

            # Get next-phrase relation and output MIDI phrases
            phrase_idx, pairs_idx = next_phrase(pm, t_bound,
                                                midi_prefix=f"{i}", save_midi=save_midi)

            phrase_idx_dict[i] = {"phrase_idx": phrase_idx,
                                  "pairs_idx": pairs_idx}

        except:
            failed[i] = row['audio_file']
            print("Phrase boundary extraciton failed", row['audio_filename'])

    # Save outputs
    raw_boundary_file = os.path.join(DATA_DIR, "raw_phrase_boundaries.json")
    with open(raw_boundary_file, "w") as f:
        json.dump(raw_phrase_boundary, f)

    boundary_file = os.path.join(DATA_DIR, "phrase_boundaries.json")
    with open(boundary_file, "w") as f:
        json.dump(phrase_boundary, f)

    failed_file = os.path.join(DATA_DIR, "phrase_boundaries_failed.json")
    with open(failed_file, "w") as f:
        json.dump(failed, f)

    index_file = os.path.join(DATA_DIR, "phrase_index.json")
    with open(index_file, "w") as f:
        json.dump(phrase_idx_dict, f)

    return


if __name__ == "__main__":

    import json
    main()
