#!/bin/bash

root_dir="../sonata-dataset"

# Convert .krn to .mxml
./humdrum/krn2xml.sh $root_dir

# Extract events as measures from .krn and .mxml file
python3 ./humdrum/event_extract.py --root_dir $root_dir

# Segment into phrases (8 measures per phrase, 2 measures as hop size) with time signature, tempo normalized; pitch tranposed to C major/minor
python3 ./humdrum/segment.py --root_dir $root_dir --len_phrase 8 --hop_size 2

# Make dataset
python3 ./krn_dataset.py --root_dir $root_dir


