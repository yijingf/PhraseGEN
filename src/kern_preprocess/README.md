# Kern Score Preprocess

## Data Cleaning
Manually remove irregular measures, correct typos and missing components. See [Appendix-Table 1](Appendix.md#table-1) for more details.

In addition, we manually complete measures at first volta brackets because Music21 is not reliable in estimating measure index after these repeat barlines. It estimates `measure.number` based on the bar line and global `measure.offset`, i.e. beat count at the beginning of the bar, thus is not reliable when measures at volta brackets are incompelete. See [Appendix-Table 2](Appendix.md#table-2) for more details.


## Preprocess Pipeline

```
bash preprocess_pipeline.sh ../../sonata-dataset
```

1. Converting to Music XML

    Convert `.krn` file to music xml ([humextra](https://github.com/craigsapp/humextra) required) because music21 is more reliable parsing `.xml` files than `.krn` files.
    ```
    bash krn2xml.sh ../../sonata-dataset
    ```

2. Music XML Parsing

    Extract music events by measures and store as JSON. 
    ```
    python3 parse_score.py
    ```

3. Phrase Segmentaion

    Segment events into 8-bar phrases and 2-bar hop size.
    ```
    python3 segment.py --len_phrase 8 --hop_size 2
    ```

4. Make Dataset
    
    Generate train/validation dataset with .
    ```
    python3 dataset.py --split_ratio 0.8 --seq_len 512 --pad_bar
    ``` 

### Phrase Annotation (Optional)

1. To ensure phrase annotation is aligned with event tokens, we first convert events to midi

    ```
    python3 render_event.py [--event path/to/event] [--output_dir output_dir]
    ```

2. Annotate the onset of the next phrase manually in Audacity and export the labels. The markers are supposed to be slightly ahead of the phrase onset.


3. Align annotation in seconds with beat bar position notation.
    ```
    python3 align_annotation.py [--label path/to/manual/annotation] [--mapping path/to/section/index/mapping] [--output path/to/output/file]
    ```
