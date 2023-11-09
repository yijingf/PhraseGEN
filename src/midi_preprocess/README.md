## Prepare MIDI Token Dataset as Control Group

To ensure that the models are compared on the same kern sonata-dataset, we preprocess the data as follows:

1. Render kern events to midi, transpose key to C major/minor, and tokenize transposed midi using Magenta's Tokenizer (Relative-timing).

    ```
    python3 midi_tokenize.py
    ```

2. Make train/validation dataset according to the same splits used for MASS model, with same sequence length, and equivalent portion of hop length between sequence. 

    ```
    python3 dataset.py --seq_len 512 --hop_size 128
    ```
