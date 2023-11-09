import os

MAESTRO_DIR = '/isi/music/yijing/maestro-v3.0.0'


curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(curr_dir))
DATA_DIR = os.path.join(root_dir, "data")


# mt3 Constants
MT3_NUM_VELOCITY_BINS = 127
MIN_PITCH = 0
MAX_PITCH = 127
STEPS_PER_SECOND = 100

# Music Transformer Constants
MT_NUM_VELOCITY_BINS = 32

# Special Token ID
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2


MT_DIRS = {"data": os.path.join(DATA_DIR, "maestro-v3.0.0"),
           "res": os.path.join(root_dir, "res", "mt"),
           "models": os.path.join(root_dir, "models", "mt"),
           "meta": os.path.join(DATA_DIR, "maestro-v3.0.0", "meta",
                                "maestro-v3.0.0.csv")}
