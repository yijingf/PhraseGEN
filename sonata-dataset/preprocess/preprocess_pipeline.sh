#!/bin/bash
#
# Kern file preprocess pipeline.
#
# Usage: bash preprocess_pipeline.sh "./"
# 

root_dir=$1

# Convert .krn to .mxml
./krn2xml.sh $root_dir

# Extract events by measures from .krn and .mxml file
python3 parse_scores.py --root_dir $root_dir

# Post-process, remove articulation, normalize irregular time signature by splitting or merging measures
python3 postparse.py

# Segment into phrases (8 measures per phrase, 2 measures as hop size) with time signature, tempo normalized; pitch tranposed to C major/minor
python3 segment.py --len_phrase 8 --hop_size 2

# Make dataset
python3 make_dataset.py --split_ratio 0.8 --seq_len 512 --pad_bar


