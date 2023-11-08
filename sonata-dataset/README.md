# Dataset Introduction

This dataset contains `.krn` files and corresponding `.midi` files of piano sonatas from 4 composers, Mozart, Beethoven, Haydn and Scarlatti during the Classical Era (1750-1825).

## Data Collection

Data were collected from [Kern Score database](https://kern.humdrum.org/cgi-bin/browse?l=/users/craig/classical).

* Mozart: https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/mozart/piano/sonata
* Beethoven: https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/beethoven/piano/sonata
* Haydn: https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/haydn/keyboard/uesonatas
* Scarlatti: https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/scarlatti/longo

`.krn` files of each composer can be downloaded as a single `.zip` file, and placed under `./krn/[composer]`.

`.midi` files and informaion of these pieces are obtained using `crawl.py`. By default, the information is stored in `./info/[composer].csv`, `midi` files are stored in `./midi/[composer]`.
