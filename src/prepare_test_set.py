import os
from fractions import Fraction

from kern_utils.tokenizer import BertTokenizer
from kern_utils.decode import decode_token_to_pm
from kern_utils.common import trim_event, load_event, normalize_ts

from midi_utils.common import change_pitch
from midi_utils.rel_tokenizer import RelTokenizer

# Constant
from kern_utils.constants import PITCH_OFFSET_DICT


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

    seg_event = {'original_time_signature': ts,
                 'key': key,
                 'original_tempo': tp}

    seg_event['original_note'] = [phrase[i]['event']
                                  for i in range(start[0], end[0] + 1)]

    seg_event['primer_note'] = [primer[i]['event']
                                for i in range(start[0], max(primer) + 1)]
    return seg_event


def main(root_dir, info, len_primer_measure=2):
    output_dir = os.path.join(root_dir, "primer_event")

    # Load Krn Tokenizer
    base_vocab_file = os.path.join(root_dir, "vocab", "base_vocab.txt")
    with open(base_vocab_file) as f:
        base_vocab = f.read().splitlines()
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    # Load Magenta's Relative MIDI-like Tokenizer
    rel_tokenizer = RelTokenizer(num_velocity_bins=1, add_eos=False)

    event_dir = os.path.join(root_dir, "event")

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
        pitch_offset = PITCH_OFFSET_DICT[orig_event['key'].split()[0]]
        change_pitch(primer_pm, pitch_shift=int(pitch_offset), inplace=True)
        primer_pm.write(os.path.join(output_dir, f"primer_{item_id}_in_C.mid"))

        # Encode primer using magenta's relative MIDI-like tokenizer
        primer_rel_tokens = rel_tokenizer.encode_pm(primer_pm)
        orig_event['primer_rel_tokens'] = primer_rel_tokens

        with open(os.path.join(output_dir, f"{item_id}.json"), "w") as f:
            json.dump(orig_event, f)


if __name__ == '__main__':
    import json
    root_dir = "../sonata-dataset"

    with open("../sonata-dataset/primer_event/info.json") as f:
        event_info = json.load(f)

    # event_info = {
    #     "01": {"event_file": "mozart/sonata09-3-1.json",
    #            "start": {"measure": 0, "pos": '3/2'},
    #            "end": {"measure": 8, "pos": '3/2'}},
    #     "02": {"event_file": "mozart/sonata09-3-1.json",
    #            "start": {"measure": 48, "pos": '0'},
    #            "end": {"measure": 55, "pos": '3'}},
    #     "03": {"event_file": "haydn/sonata61-1.json",
    #            "start": {"measure": 11, "pos": '0'},
    #            "end": {"measure": 18, "pos": '4'}},
    #     "04": {"event_file": "haydn/sonata34-1.json",
    #            "start": {"measure": 12, "pos": '3/2'},
    #            "end": {"measure": 20, "pos": '3/2'}},
    #     "05": {"event_file": "beethoven/sonata11-1.json",
    #            "start": {"measure": 4, "pos": '0'},
    #            "end": {"measure": 11, "pos": '3'}},
    #     "06": {"event_file": "beethoven/sonata09-1.json",
    #            "start": {"measure": 5, "pos": '0'},
    #            "end": {"measure": 12, "pos": '4'}},
    #     "07": {"event_file": "scarlatti/L127K348.json",
    #            "start": {"measure": 36, "pos": '0'},
    #            "end": {"measure": 43, "pos": '3'}},
    #     "08": {"event_file": "scarlatti/L166K085.json",
    #            "start": {"measure": 16, "pos": '0'},
    #            "end": {"measure": 23, "pos": '4'}},
    #     "09": {"event_file": "mozart/sonata02-1.json",
    #            "start": {"measure": 1, "pos": '0'},
    #            "end": {"measure": 8, "pos": '3'}},
    #     "10": {"event_file": "mozart/sonata15-3.json",
    #            "start": {"measure": 8, "pos": '1'},
    #            "end": {"measure": 16, "pos": '1'}},
    #     "11": {"event_file": "mozart/sonata07-1.json",
    #            "start": {"measure": 15, "pos": '0'},
    #            "end": {"measure": 22, "pos": '4'}},
    #     "12": {"event_file": "mozart/sonata15-2.json",
    #            "start": {"measure": 40, "pos": '1'},
    #            "end": {"measure": 48, "pos": '1'}},
    #     "13": {"event_file": "beethoven/sonata26-1.json",
    #            "start": {"measure": 21, "pos": '0'},
    #            "end": {"measure": 28, "pos": '4'}},
    #     "14": {"event_file": "beethoven/sonata26-3.json",
    #            "start": {"measure": 33, "pos": '0'},
    #            "end": {"measure": 40, "pos": '3'}},
    #     "15": {"event_file": "beethoven/sonata18-3.json",
    #            "start": {"measure": 0, "pos": '2'},
    #            "end": {"measure": 8, "pos": '2'}},
    #     "16": {"event_file": "beethoven/sonata24-2-0.json",
    #            "start": {"measure": 57, "pos": '1/2'},
    #            "end": {"measure": 65, "pos": '1/2'}},
    #     "17": {"event_file": "beethoven/sonata07-4.json",
    #            "start": {"measure": 73, "pos": '0'},
    #            "end": {"measure": 80, "pos": '4'}},
    #     "18": {"event_file": "scarlatti/L306K345.json",
    #            "start": {"measure": 14, "pos": '2'},
    #            "end": {"measure": 22, "pos": '2'}},
    #     "19": {"event_file": "scarlatti/L348K244.json",
    #            "start": {"measure": 15, "pos": '0'},
    #            "end": {"measure": 22, "pos": '3/2'}},
    #     "20": {"event_file": "scarlatti/L350K498.json",
    #            "start": {"measure": 40, "pos": '0'},
    #            "end": {"measure": 47, "pos": '3'}},
    #     "21": {"event_file": "beethoven/sonata24-2-0.json",
    #            "start": {"measure": 74, "pos": '3/2'},
    #            "end": {"measure": 82, "pos": '3/2'}},
    # }

    main(root_dir, event_info)
