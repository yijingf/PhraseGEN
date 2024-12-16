import os
import json
import torch
import numpy as np
from copy import deepcopy
from transformers import EncoderDecoderModel

from utils.tokenizer import decode_token_to_pm, BertTokenizer
from utils.common import normalize_tp, pitch_transpose
from utils.event import flatten_measures, map_measure_to_token

# CONSTANT
from utils.constants import PITCH_OFFSET_DICT, DATA_DIR


def reset_velocity(pm):
    for note in pm.instruments[0].notes:
        note.velocity = 75
    return


def normalize_orig_event(event, note_key='original_note'):
    # Normalize event
    ts = event['original_time_signature']
    tp = normalize_tp(event['original_tempo'])
    event['time_signature'] = f"ts-{ts}"
    event['tempo'] = f"tp-{tp}"

    # Pitch Transpose
    pitch_offset = PITCH_OFFSET_DICT[event['key'].split()[0]]
    measures = deepcopy(event[note_key])

    for i, measure in enumerate(measures):
        for j, token in enumerate(measure):
            if token[0] not in ['o', 'd']:
                measures[i][j] = pitch_transpose(token, pitch_offset)

    event['note'] = measures
    return event


class EventPreprocessor():

    def __init__(self, tokenizer, max_len=512, device=None):
        self.tokenizer = tokenizer

        self.max_len = max_len
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def get_masked_token_ids(self, orig_event, mask_measure=[2, 3, 4, 5, 6],
                             pad_bar=True):
        event = normalize_orig_event(orig_event)
        tokens = flatten_measures(deepcopy(event),
                                  eos_token=self.tokenizer.eos_token,
                                  pad_bar=pad_bar,
                                  bar_pad_token=self.tokenizer.sep_token)

        ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens))

        idxs = np.append(2, np.where(np.array(tokens) == 'bar')[0] + 1)
        idxs = np.append(idxs, len(tokens))
        mask_idx = map_measure_to_token(idxs, mask_measure)

        ids[mask_idx] = self.tokenizer.mask_id
        return ids

    def get_primer_token_ids(self, primer_event, pad_bar=False):
        event = normalize_orig_event(primer_event, note_key='primer_note')
        tokens = flatten_measures(deepcopy(event),
                                  eos_token=self.tokenizer.eos_token,
                                  pad_bar=pad_bar,
                                  bar_pad_token=self.tokenizer.sep_token,
                                  add_eos=False)

        ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens))
        return ids

    def prepare_inputs(self, token_ids, mask_pad=False):
        len_pad = self.max_len - len(token_ids)

        if len_pad < 0:
            return None

        input_ids = torch.tensor(token_ids.copy(), dtype=torch.long)
        input_ids = torch.nn.functional.pad(
            input_ids, [0, len_pad],
            value=self.tokenizer.mask_id).unsqueeze(0)

        len_primer = np.where(token_ids == self.tokenizer.mask_id)[0][0]
        decoder_input_ids = torch.tensor(token_ids[:len_primer],
                                         dtype=torch.long).unsqueeze(0)

        inputs = {"inputs": input_ids.to(self.device),
                  "decoder_input_ids": decoder_input_ids.to(self.device),
                  "max_length": self.max_len}

        return inputs


def main(model_path, test_info, output_dir,
         mask_measure=[2, 3, 4, 5, 6], max_len=512):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load Tokenizer
    base_vocab_file = os.path.join(DATA_DIR, "vocab", "base_vocab.txt")
    with open(base_vocab_file) as f:
        base_vocab = f.read().splitlines()
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)
    preprocessor = EventPreprocessor(tokenizer, max_len)

    for test_id in test_info:

        event_file = os.path.join(DATA_DIR, "primer_event", f"{test_id}.json")

        with open(event_file) as f:
            orig_event = json.load(f)

        ids = preprocessor.get_masked_token_ids(orig_event,
                                                mask_measure=mask_measure,
                                                pad_bar=True)
        inputs = preprocessor.prepare_inputs(ids)
        output = model.generate(**inputs).tolist()[0]

        pm = decode_token_to_pm(tokenizer.convert_ids_to_tokens(output))
        pm.write(os.path.join(output_dir, f"{test_id}.mid"))


if __name__ == "__main__":
    with open("../sonata-dataset/example_test_info.json") as f:
        test_info = json.load(f)

    output_dir = os.path.join(DATA_DIR, "output")
    model_path = "../models/mass"

    main(model_path, test_info, output_dir)
