# Sonata Dataset

## Description
The dataset consists of 244 piano sonata movements from four composers, Mozart, Beethoven, Haydn, and Scarlatti during the [Classical Era](https://kern.humdrum.org/cgi-bin/browse?l=/users/craig/classical). Scores are downloaded from [KernScores library](http://kern.ccarh.org/), a collection of digital music scores with Humdrum notation in **kern format. Note that for Beethoven, we only use sonatas before 1810. 

## Data Collection
1. Download scores of the following composers, and place them under `./krn/[composer]`
    * [Scarlatti](https://kern.humdrum.org/cgi-bin/ksdata?l=users/craig/classical/scarlatti/longo&format=zip)
    * [Haydn](https://kern.humdrum.org/cgi-bin/ksdata?l=users/craig/classical/haydn/keyboard/uesonatas&format=zip)
    * [Mozart](https://kern.humdrum.org/cgi-bin/ksdata?l=users/craig/classical/mozart/piano/sonata&format=zip)
    * [Beethoven](https://kern.humdrum.org/cgi-bin/ksdata?l=users/craig/classical/beethoven/piano/sonata&format=zip)
2. Optional: Get metadata of scores. Metadata will be stored in `./info/[composer].csv`,
    ```
    python3 get_meta.py
    ```

## Manual Data Cleaning
To avoid the error occurred when parsing .krn file using music21, we manually cleaned the dataset and fixed the issues that cannnot be handled by music21. See `./krn/data_cleaning.json` for these edits. **Note**: these fixes are based on the .krn files downloaded on 09/30/2024. KernScores library is still updating, these issues might not be there in the future. The current issues are categorized as follows:
1. Syntax error;
2. Extra beats within a measure, usually at the end of a section;
3. Incorrect repeating patterns;
4. Incomplete measures in the first ending brackets (music21 estimates `measure.number` based on bar line markers and beat cound at downbeat, i.e. `meausre.offset`, thus measure segmentation can be faulty if there are incomplete measures in the score.)

## Data Preprocess
See [documentation](../src/preprocess/README.md) for more details.

## Folder Structure
```
.
├── humextra                # compiled humextra tool
├── get_meta.py             # script to get meta data from KernScores library
├── parse_scores.py         # script to tokenize scores
├── segment.py              # script to obtain 8 measure segments using sliding windows
├── get_phrases.py          # script to obtain phrase token sequences #Todo
├── make_dataset.py         # script to obtain dataset for pretraining and finetuning
├── krn2xml.sh              # script to convert .krn to .xml file
├── krn                     # krn files downloaded from KernScores library
│   ├── data_cleaning.json  # manual edits to krn files for data cleaning
│   └── [composer]          # 4 subfolders, scarlatti, haydn, mozart and beethoven
|       └── [.krn files]    
├── info                    # metadata from KernScores library
│   └── [.csv files]        # 4 .csv files of metadata for sonatas of each composer
├── mxml                    # music xml files converted from .krn files
│   └── [composer]          # 4 subfolders, scarlatti, haydn, mozart and beethoven
|       └── [.xml files]    
├── event                   # tokenized scores
│   └── [composer]          # 4 subfolders, scarlatti, haydn, mozart and beethoven
|       └── [.json files]   
├── segment                 # tokenized and normalized score segments
│   └── [composer]          # 4 subfolders, scarlatti, haydn, mozart and beethoven
|       └── [.json files]   
├── dataset                 # train/validation dataset
│   ├── train_val_split.csv # Train/validation split configuration
│   ├── train_512_pad.json  # Trainining set
│   └── val_512_pad.json    # Validation set
├── vocab                   # vocabulary
│   └── base_vocab.txt      
├── boundary_predictions    # boundary predicted by phrase segmentation algorithm
│   └── [.pkl files]        
├── phrase                  # phrase token sequences
│   └── [composer]          # 4 subfolders, scarlatti, haydn, mozart and beethoven
|       └── [.json files]  
└── ...
```