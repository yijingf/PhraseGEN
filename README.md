# PhraseGEN: Phrase-Level Symbolic Music Generation

This is the code repository for our paper [Phrase-Level Symbolic Music Generation](https://aimc2024.pubpub.org/pub/2e7q3cpr).

## Dataset
You may download the digital scores in **kern** format and prepare the **Sonata Dataset** used in the paper, or prepare your own dataset.

### Prepare the Sonata Dataset
1. Data collection: download the musical scores in the Humdrum **kern data format. See `sonata-dataset/README.md` for more details on the classical piano sonata collected in this dataset. 
2. Preprocess: See `./src/preprocess/README.md` for more details.

## Training
    ```
    python3 train.py
    ```

## Inference
1. Prepare test set with the test samples in `sonata-dataset/example_test_info.json`.
    ```
    python3 prepare_testset.py
    ```
2. Inference
    ```
    python3 inference.py
    ```

## Citation
If you find this repo useful, please cite our paper.

```
@article{Feng2024Phrase,
	author = {Feng, Yijing and Sahin, Egemen and Casey, Michael A.},
	journal = {AIMC 2024 (09/09 - 11/09 )},
	year = {2024},
	month = {aug 29},
	note = {https://aimc2024.pubpub.org/pub/2e7q3cpr},
	publisher = {},
	title = {Phrase-{Level} {Symbolic} {Music} {Generation}},
}
```

