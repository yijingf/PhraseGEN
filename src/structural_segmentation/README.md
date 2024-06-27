# Structural Segmentation
We extract the structural graph of sonata music through a combination of phrase segmentation, music form anaylsis and key analysis. 

Phrase boundaries are detected from audio in an unsupervised manner, by grouping similar patches of audio descriptors boosted with a triplet mining method. 

The analysis of music form is rule-based. We estimate section boundaries indicated by score notations (such as repeat signs and changes in key and time signatures), and refine the phrase boundaries detected above.

Key analysis is done using Music21. 


## Phrase Segmentation
See `README.md` in [PhraseSegmentation](https://git.dartmouth.edu/f004kkq/phrasesegmentation) repo for more details. 
Phrase segmentation output are stored as `.pkl` files containing pairs of timing of predicted boundaries and the cluster ids corresponding to the intervals.

## Music Form Analysis

### Music form from score

#### Usage
```
import json
from struct_segment import main

composer, file_base = "beethoven", "sonata08-1"
sect_phrase = main(composer, file_base)

with open("example_struct.json", "w") as f:
    json.dump(sect_phrase, f)
```

The capital letter notations in the `.krn` files notate the materials between two repeat-signs, which is refered to as "score section". We use different heuristics to estimate the boundaries of large section, refered to as "section", i.e. Exposition, Development and Recapitulation, depending on the pattern of these notations. The patterns are categorized as follows:

#### X
There isn't any repeat-signs in the middle of a movement. We don't assume such movements would follow the Expostion-Development-Recapitulation format, and therefore identify the section boundaries by occurences of elements identical to the first phrase. 

#### XY
We assume that there is either a recapitulation in Y or no recapitulation; development starts from Y. XY type is usually presented as 'AB' on score.

#### XYZ
The XYZ type covers movements with more than three score sections, such as 'ABCD', 'ABCDE', etc. We assume that the recapitulation can be found in sections after B or there is no recapitulation.

#### XYX
The second X is an exact repeat of the first X. The XYX type is usually presented as 'AB-x-AB' on score, where x can be any materials that are different from 'AB'. In this case, 'AB' and 'x' match exposition and development respectively.

#### XYX'
The second X' is a variant of the first X, with the first phrase being identical. The XYX' type can be presented as 'AB-x-ABy' on score. In this case, exposition, developement, and recapitulation would match 'AB', 'x' and 'ABy', respectively.

#### I-x
If the score section notation starts with 'I', we assume that the I section corresponds to an Introduction. The segmentation rules described above apply to the remaining material.

### Types of phrase relation
Given the section-phrase structure identified above, we define three types of between-phrase relationship: 
1. **progression**: between phrases within same section
