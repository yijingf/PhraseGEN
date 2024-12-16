import os
from fractions import Fraction

from utils.tokenizer import BertTokenizer
from utils.decode import decode_token_to_pm
from utils.common import trim_event, load_event, normalize_ts, pm_pitch_transpose

# Constant
from utils.constants import PITCH_OFFSET_DICT, DATA_DIR


def check_consistency(phrase):
    # sanity check
    tempo, time_signature, key = set(), set(), set()

    for i in phrase:
        tempo.add(phrase[i]['tempo'])
        time_signature.add(phrase[i]['time_signature'])
        key.add(phrase[i]['key'])

    if any([len(key) > 1, len(tempo) > 1, len(time_signature) > 1]):
        raise ValueError("Inconsistent key, tempo or time signature")

    return


def get_orig_event(event_file, start=(0, 0), end=(8, 4), len_primer_measure=2):
    event, _ = load_event(event_file)
    phrase = trim_event(event, start, end)

    primer_end = (start[0] + len_primer_measure, start[1])
    primer = trim_event(event, start, primer_end)

    check_consistency(phrase)

    tp = phrase[start[0]]['tempo']
    key = phrase[start[0]]['key']
    ts = phrase[start[0]]['time_signature']

    seg_event = {'original_time_signature': ts, 'key': key, 'original_tempo': tp}

    seg_event['original_note'] = [phrase[i]['event']
                                  for i in range(start[0], end[0] + 1)]

    seg_event['primer_note'] = [primer[i]['event']
                                for i in range(start[0], max(primer) + 1)]
    return seg_event


def main(info, event_dir, output_dir, len_primer_measure=2):

    for item_id, item in info.items():

        event_file = os.path.join(event_dir, item['event_file'])
        start = (item["start"]['measure'], Fraction(item['start']['pos']))
        end = (item["end"]['measure'], Fraction(item['end']['pos']))

        orig_event = get_orig_event(event_file, start, end, len_primer_measure)

        # Decode original midi
        norm_ts = normalize_ts(orig_event['original_time_signature'])
        midi_tp = int(orig_event['original_tempo'] / 12) * 12

        tokens = [f'ts-{norm_ts}', f'tp-{midi_tp}']
        for measure_token in orig_event['original_note']:
            tokens += measure_token
            tokens += ['bar']

        if tokenizer.has_irregular_token(tokens[2:]):
            raise ValueError("Irregular tokens")

        # Decode
        pm = decode_token_to_pm(tokens)
        pm.write(os.path.join(output_dir, f"{item_id}.mid"))

        # Decode primer
        primer_tokens = [f'ts-{norm_ts}', f'tp-{midi_tp}']
        for measure_token in orig_event['primer_note']:
            primer_tokens += measure_token
            primer_tokens += ['bar']
        primer_pm = decode_token_to_pm(primer_tokens)

        # Pitch transpose to C Major/Minor
        pitch_offset = int(PITCH_OFFSET_DICT[orig_event['key'].split()[0]])
        pm_pitch_transpose(primer_pm, pitch_shift=pitch_offset, inplace=True)
        primer_fname = os.path.join(output_dir, f"primer_{item_id}_in_C.mid")
        primer_pm.write(primer_fname)

        with open(os.path.join(output_dir, f"{item_id}.json"), "w") as f:
            json.dump(orig_event, f)


if __name__ == '__main__':
    import json
    with open("../sonata-dataset/example_test_info.json") as f:
        event_info = json.load(f)

    event_dir = os.path.join(DATA_DIR, "event")
    output_dir = os.path.join(DATA_DIR, "primer_event")

    # Load Tokenizer
    base_vocab_file = os.path.join(DATA_DIR, "vocab", "base_vocab.txt")
    with open(base_vocab_file) as f:
        base_vocab = f.read().splitlines()
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    main(event_info, event_dir, output_dir)
