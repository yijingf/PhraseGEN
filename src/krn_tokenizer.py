import json
import warnings
import pretty_midi
import numpy as np
from fractions import Fraction


default_time_signature = '4/4'
default_tempo = 120
default_velocity = 100


def get_time(event):
    return Fraction(event.split('-')[-1])


def pitch_name_to_pm_pitch(pitch_name):

    if '-' in pitch_name:
        pitch_name = "".join(pitch_name.split('-'))
        pitch = pretty_midi.note_name_to_number(pitch_name) - 1
    else:
        pitch = pretty_midi.note_name_to_number(pitch_name)

    return pitch


class BaseTokenizer():

    def __init__(self):

        self.pad_token = 'pad'
        self.eos_token = 'eos'
        self.unk_token = 'unk'

        self.special_tokens = ['pad', 'eos', 'unk']

        self.pad_id = 0
        self.eos_id = 1
        self.unk_id = 2

        self.special_token_ids = [0, 1, 2]
        self.n_special_token = 3

        self.vocab_size = 0
        self.token_to_id = {}
        self.id_to_token = {}

    def load_vocab(self, vocab_file):
        with open(vocab_file) as f:
            tokens = f.read().splitlines()

        ids = list(range(len(tokens)))
        self.token_to_id = dict(zip(tokens, ids))
        self.id_to_token = dict(zip(ids, tokens))

        # Sanity Check
        for token in self.special_tokens:
            assert token not in self.token_to_id, f"Token {token} not in vocab."
            assert self.token_to_id[token] == getattr(self, f"{token}_id")

        assert tokens[self.n_special_token - 1] in self.special_tokens
        self.vocab = tokens
        self.vocab_size = len(tokens)
        return

    def save_vocab(self, vocab_file):
        with open(vocab_file, "w") as f:
            for token in self.vocab:
                f.write(token + "\n")

    def load_vocab_json(self, vocab_json_file):
        with open(vocab_json_file) as f:
            self.token_to_id = json.load(f)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Sanity Check
        for token in self.special_tokens:
            assert token in self.token_to_id, f"Token {token} not in vocab."
            assert self.token_to_id[token] == getattr(self, f"{token}_id")

        self.vocab_size = len(self.id_to_token)
        self.vocab = [self.id_to_token[i] for i in range(self.vocab_size)]
        return

    def save_vocab_json(self, vocab_json_file):
        with open(vocab_json_file, "w") as f:
            json.dump(self.token_to_id, f)
        return

    def train(self, tokens):
        tokens = sorted(set(tokens))

        for token in self.special_tokens:
            if token in tokens:
                tokens.remove(token)

        self.vocab = self.special_tokens + tokens
        self.vocab_size = len(self.vocab)
        ids = list(range(self.vocab_size))

        self.token_to_id = dict(zip(self.vocab, ids))
        self.id_to_token = dict(zip(ids, self.vocab))

        return

    def add_special_token(self, token):
        if token in self.special_tokens:
            Warning(f"Token: {token} is already a special token.")
            return

        token_attr = f"{token}_token"
        setattr(self, token_attr, token)

        self.special_tokens += [token]

        id_attr = f"{token}_id"
        setattr(self, id_attr, self.n_special_token)
        self.special_token_ids += [self.n_special_token]
        self.n_special_token += 1

        return

    def convert_tokens_to_ids(self, tokens):
        token_ids = [self.token_to_id[i] for i in tokens]
        return token_ids

    def convert_ids_to_tokens(self, token_ids):
        tokens = [self.id_to_token[i] for i in token_ids]
        return tokens

    def postprocess(self, token_ids):

        # Trim padding or eos token
        processed_token_ids = []
        for token_id in token_ids:
            if token_id in [self.eos_id, self.pad_id]:
                break
            else:
                processed_token_ids.append(token_id)

        return token_ids


class MTTokenizer(BaseTokenizer):

    def __init__(self, vocab_file=None, vocab_json_file=None):

        super().__init__()

        if vocab_file:
            self.load_vocab(vocab_file)

        if vocab_json_file:
            self.load_vocab_json(vocab_json_file)

        return

    def decode(self, token_ids):

        token_ids = self.postprocess(token_ids)
        tokens = self.convert_ids_to_tokens(token_ids)

        pm = decode_token_to_pm(tokens, bar_eos_token=None)

        return pm


class BertTokenizer(BaseTokenizer):

    def __init__(self, vocab_file=None, vocab_json_file=None):

        super().__init__()
        self.add_special_token('sep')  # used as end of bar token
        self.add_special_token('mask')

        if vocab_file:
            self.load_vocab(vocab_file)

        if vocab_json_file:
            self.load_vocab_json(vocab_json_file)

        return

    def decode(self, token_ids, bar_eos_token=None):

        token_ids = self.postprocess(token_ids)
        tokens = self.convert_ids_to_tokens(token_ids)

        if not bar_eos_token:
            bar_eos_token = self.sep_token
        pm = decode_token_to_pm(tokens, bar_eos_token)

        return pm


def get_ts_tp(tokens):

    ts, tp = None, None

    for token in tokens:

        if token[:2] == 'ts':
            ts = token[3:]
        elif token[:2] == 'tp':
            tp = int(token[3:])

        if ts and tp:
            break

    return ts, tp


def decode_token_to_event(tokens, bar_eos_token='eos'):
    ts, tp = get_ts_tp(tokens)
    ts_denom = Fraction(ts).denominator

    events = {}
    notes = []

    onset = None
    bar = 0
    events[bar] = []

    seq_len = len(tokens)

    i = 0
    decode_flag = True

    while i < seq_len:
        token = tokens[i]
        i += 1

        if token == 'bar':
            decode_flag = True
            onset = Fraction(0)

            for note in notes:
                dur = ts_denom - note[0]
                events[bar].append(note + [dur])

            bar += 1
            events[bar] = []
            notes = []
            continue

        if not decode_flag:
            continue

        if token == bar_eos_token:
            decode_flag = False
            continue

        elif token[:2] in ['ts', 'tp']:
            continue

        elif token[0] == 'o':
            onset = get_time(token)

        elif token[0] == 'd':
            dur = get_time(token)
            for note in notes:
                events[bar].append(note + [dur])

            notes = []

        else:
            notes.append([onset, token])

    return ts, tp, events


def decode_token_to_pm(tokens, bar_limit=False):

    ts, tp, events = decode_token_to_event(tokens)

    if not ts:
        warnings.warn(f"No time signature. Set to {default_time_signature}")
        ts = default_time_signature

    if not tp:
        warnings.warn(f"No tempo. Set to {default_tempo}")
        tp = default_tempo

    t_quarter_note = 60 / tp

    ts_digit = ts.split('/')
    if len(ts_digit) > 2:
        ts_num = Fraction(ts_digit[0], ts_digit[1])
    else:
        ts_num = int(ts_digit[0])
    ts_denom = int(ts_digit[-1])

    assert ts_denom == 4

    inst = pretty_midi.Instrument(program=0)
    n_bar = len(events)
    for i in range(n_bar):
        for onset, pitch_name, duration in events[i]:

            if bar_limit and onset >= ts_num:
                continue

            t_st = (onset + i * ts_num) * t_quarter_note
            t_ed = t_st + duration * t_quarter_note
            pitch = pitch_name_to_pm_pitch(pitch_name)
            note = pretty_midi.Note(default_velocity, pitch, t_st, t_ed)
            inst.notes.append(note)

    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(inst)
    return pm
