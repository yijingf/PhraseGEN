#!/bin/bash
#
# Kern file preprocess pipeline.
#
# Usage: bash preprocess_pipeline.sh "../../sonata-dataset"
# 

root_dir=$1

# Convert .krn to .mxml
./krn2xml.sh $root_dir

# Extract events by measures from .krn and .mxml file
python3 parse_score.py --root_dir $root_dir

# Segment into phrases (8 measures per phrase, 2 measures as hop size) with time signature, tempo normalized; pitch tranposed to C major/minor
python3 segment.py --root_dir $root_dir --len_phrase 8 --hop_size 2

# Make dataset
python3 dataset.py --root_dir $root_dir --split_ratio 0.8 --seq_len 512 --pad_bar


