# PhraseGEN: Phrase-Level Symbolic Music Generation

This is the code repository for our paper [Phrase-Level Symbolic Music Generation](https://aimc2024.pubpub.org/pub/2e7q3cpr).

## Dataset
You may download the digital scores in **kern** format and prepare the **Sonata Dataset** used in the paper, or prepare your own dataset.

### Prepare the Sonata Dataset
1. Download the musical scores in the Humdrum **kern data format. See `./sonata-dataset/README.md` for more details on the classical piano sonata collected in this dataset. 
2. Run the preprocess pipeline.
3. Optional: prepare musical phrases for fine-tuning.

### Prepare Your Own Dataset
Musical phrases generation.
* Phrase segmentation. `./src/segment.py`
* MIDI tokenization and prepare the dataset. `./src/preprocess.py`

## Inference


## Training


## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

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

