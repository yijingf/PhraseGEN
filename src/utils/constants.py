import os
import numpy as np
from music21 import pitch

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(curr_dir))

DATA_DIR = os.path.join(root_dir, "sonata-dataset")
EVENT_DIR = os.path.join(DATA_DIR, "event")
MIDI_DIR = os.path.join(DATA_DIR, "midi")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
UNROLL_EVENT_DIR = os.path.join(DATA_DIR, "unrolled_event")

# Build key transpose mapping
# C#4 to G4 -> C4; G#3 to B3 -> C4
PITCH_OFFSET_DICT = {}

base_ps = pitch.Pitch('C4').ps
pitch_pivot = base_ps + 12 / 2

for key in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
    for acc in ['', '-', '#']:

        ks = f"{key}{acc}"

        ks_ps = pitch.Pitch(ks).ps
        if ks_ps > pitch_pivot:
            ks_ps -= 12

        PITCH_OFFSET_DICT[ks] = base_ps - ks_ps

# Bins of regular tempi
TEMPO_BIN = np.array([24, 40, 60, 72, 96, 120, 144, 160, 192, 200])
