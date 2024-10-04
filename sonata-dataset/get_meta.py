"""
Get classical piano sonata metadata from KernScores Library and download corresponding midi files (optional). Metadata are stored as `./info/[composer].csv`; midi files, if downloaded, are stored in `./midi/[composer]`.

Usage: 
python3 get_meta.py or python3 get_meta.py --download_midi
"""

import os
import urllib
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from bs4.element import NavigableString

INFO_DIR = "./info"
MIDI_DIR = "./downloaded_midi"

COMPOSER_URLS = {
    "mozart": "https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/mozart/piano/sonata",
    "beethoven": 'https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/beethoven/piano/sonata',
    "haydn": "https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/beethoven/piano/sonata", "scarlatti": "https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/scarlatti/longo"
}


def normalize(items):
    normalized = []
    for item in items:
        if item.name is None:
            normalized.append(item)
        elif item.name == 'hlend':
            for i in item:
                if type(i) is NavigableString:
                    normalized.append(i)
                elif i.name == 'tt':
                    normalized.append(i.text)
        else:
            continue
    normalized = ''.join(normalized)
    normalized = normalized.replace('[]', '')
    normalized = normalized.replace('\xa0', '')
    return normalized.lstrip().strip()


def crawl_scarlatti(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    res = soup.find_all('tr', class_='bbb33')

    title_lst = []
    href_lst = []

    for item in res:
        href = item.find('a', href=lambda text: "format=midi" in text)
        title = item.find('td', class_='aaa49')

        if title is None:
            continue

        title_lst.append(title.text.replace('\xa0', ''))
        href_lst.append(href['href'])

    df = pd.DataFrame({'title': title_lst, 'url': href_lst})
    df['filename'] = [url.split('.krn')[0].split("file=")[-1]
                      for url in df['url']]
    return df


def crawl(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    # Figure out class number
    title_block = soup.find(
        text=lambda text: "Sonata No." in text).parent.parent

    while title_block.name != 'tr':
        title_block = title_block.parent
    # title_block = soup.find('hlstart').parent.parent
    title_class = title_block['class'][0]

    mov_block = soup.find(
        'a', href=lambda text: "format=midi" in text).parent.parent
    mov_class = mov_block['class'][0]

    title_lst = []
    movs_lst = []
    href_lst = []

    all_entries = title_block.parent
    while all_entries.find('a', href=lambda text: "format=midi" in text) is None:
        all_entries = all_entries.parent

    for item in all_entries:

        if type(item) == NavigableString:
            continue

        cls = item['class'][0]

        # Title
        if cls == title_class:
            entries = item.find(text=lambda text: "Sonata No." in text).parent
            title = normalize(entries.contents)

        # Movement
        elif cls == mov_class:

            try:
                # Get MIDI href
                href = item.find('a',
                                 href=lambda text: "format=midi" in text)['href']

                # Get movement title
                mov_title = item.text.replace("\xa0", "")
            except:
                continue

            title_lst.append(title)
            movs_lst.append(mov_title)
            href_lst.append(href)

    df = pd.DataFrame({"title": title_lst, "mov.": movs_lst, 'url': href_lst})
    df['filename'] = [url.split('.krn')[0].split("file=")[-1]
                      for url in df['url']]

    return df


def fetch_midi(df, midi_dir):
    """Loop through MIDI urls in df dataframe and download them.

    Args:
        df (pandas.Dataframe)
        midi_dir (str)
    """
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(midi_dir, f"{row['filename']}.mid")
        urllib.request.urlretrieve(row['url'], path)
    return


def main(download_midi=True):

    os.makedirs(INFO_DIR, exist_ok=True)
    if download_midi:
        os.makedirs(MIDI_DIR, exist_ok=True)

    # Get metadata for Mozart, Beethoven, Haydn
    for composer in ['mozart', 'beethoven', 'haydn']:
        url = COMPOSER_URLS[composer]

        # Crawl MIDI urls and other info into a dataframe.
        df = crawl(url)
        filename = os.path.join(INFO_DIR, f'{composer}.csv')
        df.to_csv(filename, index=False)

        if download_midi:
            midi_dir = os.path.join(MIDI_DIR, composer)

            if os.path.exists(midi_dir):
                print(f"{midi_dir} already exists.")
            else:
                os.makedirs(midi_dir)
            fetch_midi(df, midi_dir)

    # Get metadata for Mozart, Beethoven, Haydn
    url = 'https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/scarlatti/longo'
    df = crawl_scarlatti(url)
    filename = os.path.join(INFO_DIR, 'scarlatti.csv')
    df.to_csv(filename, index=False)

    if download_midi:
        fetch_midi(df, midi_dir)

    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download_midi", dest="download_midi", default="store_true",
                        help="Download midi files.")
    args = parser.parse_args()

    main(args.download_midi)
