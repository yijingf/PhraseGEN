""" Modulize Magenta's MT3 MIDI (note sequence) tokenization.

Example:
```
    import note_seq
    from tokenizer import MT3Tokenizer

    ns = note_seq.NoteSequence()
    ns.notes.add(start_time=1.0, end_time=3.0,
                 pitch=70, velocity=1)

    ns.notes.add(start_time=0.5, end_time=4.0,
                 pitch=62, velocity=127)
    ns.total_time = ns.notes[-1].end_time

    tokenizer = MT3Tokenizer()

    tokens = tokenizer.encode(ns)
    decoded_ns = tokenizer.decode(tokens)
```

"""
import note_seq
import numpy as np

import vocabularies
import note_sequences
import run_length_encoding

from constants import PAD_ID, EOS_ID, UNK_ID
from constants import STEPS_PER_SECOND, MT3_NUM_VELOCITY_BINS


def trim_eos(tokens):
    """
    For only one sequence!
    """
    tokens = np.array(tokens, np.int32)
    if vocabularies.DECODED_EOS_ID in tokens:
        tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
    return tokens


class MT3Tokenizer():

    def __init__(self, num_velocity_bins=MT3_NUM_VELOCITY_BINS,
                 steps_per_second=STEPS_PER_SECOND,
                 pad_id=PAD_ID, eos_id=EOS_ID, unk_id=UNK_ID, add_eos=True):

        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins,
                                                       steps_per_second=steps_per_second))
        self.encoding_spec = note_sequences.NoteEncodingSpec

        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id

        self.num_special_tokens = 3
        self.vocab_size = 1517

        self.add_eos = add_eos

        # self.cls_id = self.vocab_size
        # self.sep_id = self.vocab_size + 1
        # self.mask_id = self.vocab_size + 2

    def rle_shift(self, events):
        """Run length encoding: compress time shift tokens.

        Args:
            events (list): _description_

        Returns:
            list: _description_
        """
        shift_steps, total_shift_steps = 0, 0
        output = []

        for event in events:
            if self.codec.is_shift_event_index(event):
                shift_steps += 1
                total_shift_steps += 1

            else:
                if shift_steps > 0:
                    shift_steps = total_shift_steps
                    while shift_steps > 0:
                        output_steps = min(
                            self.codec.max_shift_steps, shift_steps)
                        output.append(output_steps)
                        shift_steps -= output_steps
                output.append(event)

        return output

    def encode(self, ns):
        """Tokenize note sequence.

        Returns:
            a list of token ids
        """
        event_times, event_values = (
            note_sequences.note_sequence_to_onsets_and_offsets(ns))

        frame_times = np.arange(
            0, ns.total_time, step=1 / self.codec.steps_per_second)

        events, _, _, _, _ = run_length_encoding.encode_and_index_events(
            state=None,
            event_times=event_times,
            event_values=event_values,
            encode_event_fn=note_sequences.note_event_data_to_events,
            codec=self.codec,
            frame_times=frame_times)

        tokens = self.rle_shift(events)

        if self.add_eos:
            tokens.append(self.eos_id)
        return tokens

    def encode_pm(self, pm):
        """Tokenize pretty midi. 

        Returns:
            a list of token ids
        """
        ns = note_seq.midi_to_note_sequence(pm)
        return self.encode(ns)

    def decode(self, tokens, start_time=0):
        """
        Returns:
            _type_: _description_
        """
        assert isinstance(tokens, list) or isinstance(
            tokens, np.ndarray), "Decoder only handles list or numpy.ndarray"

        decoding_state = self.encoding_spec.init_decoding_state_fn()

        invalid_ids, dropped_events = run_length_encoding.decode_events(
            state=decoding_state,
            tokens=tokens,
            start_time=start_time, max_time=None,
            codec=self.codec,
            decode_event_fn=self.encoding_spec.decode_event_fn)

        # ns = note_sequences.flush_note_decoding_state(decoding_state)
        ns = self.encoding_spec.flush_decoding_state_fn(decoding_state)

        return ns, invalid_ids, dropped_events

    def postprocess(self, ids):
        # Todo: fix unknown index post processing.

        # Replace ids > base_vocab_size with unk_id (unknown id).
        ids = np.where(np.less(ids, self.base_vocab_size), ids, self.unk_id)

        # Replace everything after the first eos_id with pad_id.
        equal = (ids == self.eos_id)
        # shift equal to exclude the first eos_id
        equal = np.pad(equal[:, :-1], ((0, 0), (1, 0)),
                       'constant', constant_values=False)
        after_eos = np.cumsum(equal, axis=-1).astype(bool)

        ids = np.where(after_eos, self.pad_id, ids)

        eos_and_after = np.cumsum(ids == self.eos_id, axis=-1).astype(bool)

        ids = np.where(eos_and_after,
                       vocabularies.DECODED_EOS_ID,
                       np.where(
                           np.logical_and(
                               np.greater_equal(ids, self.num_special_tokens),
                               np.less(ids, self.base_vocab_size)),
                           ids - self.num_special_tokens,
                           vocabularies.DECODED_INVALID_ID))

        return ids


if __name__ == "__main__":

    ns = note_seq.NoteSequence()
    ns.notes.add(start_time=1.0, end_time=3.0,
                 pitch=70, velocity=1)

    ns.notes.add(start_time=0.5, end_time=4.0,
                 pitch=62, velocity=127)
    ns.total_time = ns.notes[-1].end_time

    tokenizer = MT3Tokenizer()

    tokens = tokenizer.encode(ns)
    decoded_ns = tokenizer.decode(tokens)

    # Todo: unit test
    note_seq.note_sequence_to_midi_file(decoded_ns[0], 'output.mid')

    # predictions = [postprocess(events)]
    # decoded_ns = event_predictions_to_ns(predictions, codec, encoding_spec)
