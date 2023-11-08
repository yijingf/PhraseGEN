import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(curr_dir))


if 'dartfs-hpc' in curr_dir:
    # on discovery
    DATA_DIR = '/isi/music/yijing'
else:
    DATA_DIR = os.path.join(root_dir, "data")


# mt3 tokenization
NUM_VELOCITY_BINS = 127


MT_DIRS = {"data": os.path.join(DATA_DIR, "maestro-v3.0.0"),
           "res": os.path.join(root_dir, "res", "mt"),
           "models": os.path.join(root_dir, "models", "mt"),
           "meta": os.path.join(DATA_DIR, "maestro-v3.0.0", "meta",
                                "maestro-v3.0.0.csv")}
