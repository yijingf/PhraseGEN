# Data Preprocess

Before preprocessing, kern files are first manually screened and cleaned. Find more details [here](../../sonata-dataset/README.md). 

kern files are first parsed using [music21](https://pypi.org/project/music21/). Although music21 can parse .krn files directly, it has more unknown errors when loading .krn files than music .xml files. Therefore, converting .krn to .xml is optional but recommended.


### Build Pretrain Dataset

1. Reformatting

    Convert .krn files to .mxml files using `hum2xml` from `humextra` tool. See humextra [github repository](https://github.com/humdrum-tools/humextra) for more details on downloading, compiling and installing. Make sure `humextra` is in this directory. The outputs will be redirect to `../../sonata-dataset/mxml`.
    ```
    bash krn2xml.sh ../../sonata-dataset/krn
    ```

2. Tokenization

    Extract notes (grouped by measures), tempo, time signature from all the kern files. The outputs will be redirect to `../../sonata-dataset/event`.
    ```
    python3 parse_scores.py
    ```

3. Segmentation

    This script segments the movements into 8 measures segments with a hop size of 2 measures, normalizes time signature and tempo, and transpose pitches to C major/minor. The outputs will be redirect to `../../sonata-dataset/segment`.
    ```
    python3 segment.py --len_phrase 8 --hop_size 2
    ```

4. Build Pretraining Dataset

    Make masked, bar-level padded dataset for training PhraseGEN model. The dataset and split config will be redirected to `../../sonata-dataset/dataset`; the vocabluary will be redirected to `../../sonata-dataset/vocab/base_vocab.txt`. 
    ```
    # split train/validation dataset by 0.8/0.2, generate bar-level padded sequence with a maximum length of 512

    python3 make_dataset.py --split_ratio 0.8 --seq_len 512 --pad_bar

    ```

### Build Fine-tune

This step extracts "real" phrases as a result of auto segmentation for finetuning the model. 

1. Render
    
    Expand the repeats in score to produce midi files, and render midi to audio accordingly.    
    ```
    python3 render_event.py --repeat_mode "no_repeat"
    ```

2. Phrase Segmentation
    
    * Perform [automatic phrase segmentation](https://github.com/yijingf/Phrase-Segmentation). Clone the repository in another directory. Make sure to move and rename all the rendered midi files to `path/to/phrase/segmentation/repo/data/midi` and index mapping .json files to `path/to/phrase/segmentation/repo/data/info`. Follow the steps to train and obtain phrase boundary predictions in time domain. Redirect the phrase boundary predictions to
    `../../sonata-dataset/boundary_predictions`.

    * Convert phrase boundary markers in time to measure/beat representation and get the token sequences for these phrases. 
        ```
        python3 phrase_segment.py
        ```

    * Make fintune dataset following the same train/validation split configuration.
        ```
        python3 make_dataset.py --data_dir DATA_DIR/phrase
        ```

