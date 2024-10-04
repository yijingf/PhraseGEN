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

## Data Parsing
.krn files are first parsed using [music21](https://pypi.org/project/music21/). Although music21 can parse .krn files directly, it has more unknown errors when loading .krn files than music .xml files. Therefore, converting .krn to .xml is optional but recommended.

1. Convert .krn files to .mxml files using `hum2xml` from `humextra` tool. See humextra [github repository](https://github.com/humdrum-tools/humextra) for more details on downloading, compiling and installing. Make sure `humextra` is in the `[root]/sonata-dataset` directory. The outputs will be redirect to `./mxml`.
```
# process all .krn files
bash krn2xml.sh ./krn
```
2. Tokenization. Extract notes (grouped by measures), tempo, time signature from kern file. The outputs will be redirect to `./event`.
```
python3 parse_scores.py
```
3. Sliding window segmentation. This script segments the movements into 8 measures segments with a hop size of 2 measures, normalize time signature and tempo, and transpose pitches to C major/minor. The outputs will be redirect to `./segment`.
```
python3 segment.py --len_phrase 8 --hop_size 2
```
4. Make masked, bar-level padded dataset for training PhraseGEN model. The dataset and split config will be redirected to `./dataset`; the vocabluary will be redirected to `./vocab/base_vocab.txt`. 
```
# split train/validation dataset by 0.8/0.2, generate bar-level padded sequence with a maximum length of 512
python3 make_dataset.py --split_ratio 0.8 --seq_len 512 --pad_bar
```
5. Phrase segmentation. This step outputs "real" phrases as a result of auto segmentation for finetuning the model. 
    * Unroll repeats in score for rendering midi files.
    ```
    python3 render_event.py --repeat_mode "no_repeat"
    ```
    * Perform [automatic phrase segmentation](https://github.com/yijingf/Phrase-Segmentation). Clone the repository in another directory. Make sure to move and rename all the rendered midi files to `path/to/phrase/segmentation/repo/data/midi` and index mapping .json files to `path/to/phrase/segmentation/repo/data/info`. Follow the steps to train and obtain phrase boundary predictions in time domain. Redirect the phrase boundary predictions to `./boundary_predictions`.
    * Convert phrase boundary markers in time to measure/beat representation and get the token sequences for these phrases. 
        ```
        python3 get_phrases.py # Todo
        ```
    * Make fintune dataset following the same train/validation split configuration.
        ```
        python3 make_dataset.py --phrase # Todo
        ```

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