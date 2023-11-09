""" Modulize Magenta's Music Transformer MIDI (note sequence) tokenization.

"""

import note_seq
from music_encoders import MidiPerformanceEncoder

from constants import EOS_ID, PAD_ID
from constants import MIN_PITCH, MAX_PITCH, STEPS_PER_SECOND, MT_NUM_VELOCITY_BINS


class RelTokenizer():

    def __init__(self, num_velocity_bins=MT_NUM_VELOCITY_BINS,
                 steps_per_second=STEPS_PER_SECOND,
                 min_pitch=MIN_PITCH, max_pitch=MAX_PITCH,
                 eos_id=EOS_ID, pad_id=PAD_ID, add_eos=True):

        self.mpe = MidiPerformanceEncoder(steps_per_second=steps_per_second,
                                          num_velocity_bins=num_velocity_bins,
                                          min_pitch=min_pitch,
                                          max_pitch=max_pitch,
                                          add_eos=add_eos)

        # Number of tokens in magenta encoding + padding + EOS
        self.vocab_size = self.mpe.unigram_vocab_size  # + 2?
        self.eos_id = eos_id
        self.pad_id = pad_id

    def encode_pm(self, pm):
        """Returns a sequence of magenta performance encoding.

        Args:
            midi (pretty_midi.PrettyMIDI): Single input midi in pretty_midi.PrettyMIDI format.

        Returns:
            list: magenta performance encoding.
        """
        seq_proto = note_seq.midi_to_sequence_proto(pm)
        tokens = self.mpe.encode_note_sequence(seq_proto)
        return tokens

    def decode(self, ids):
        """Decode magenta performance indices corresponding to event n-grams back into the n-grams,
        and convert to pretty_midi. (Adapted from magenta's update in Jan. 2022, will use magenta's own decode_to_note_sequence function later)

        Args:
            ids (list): Magenta performance encodings.

        Returns:
            pretty_midi.PrettyMIDI: decoded midi in pretty_midi.PrettyMIDI format.
        """

        # def magenta_decode_midi(notes, is_eos=False):
        # mpe = MidiPerformanceEncoder(
        #     steps_per_second=STEPS_PER_SECOND,
        #     num_velocity_bins=NUM_VELOCITY_BINS,
        #     min_pitch=MIN_PITCH,
        #     max_pitch=MAX_PITCH,
        #     add_eos=is_eos)
        # # pm = mpe.decode(notes, return_pm=True)
        # pm = mpe.decode(notes)
        # return pm

        event_ids = []
        for i in ids:
            if i >= self.mpe.unigram_vocab_size:
                event_ids += self.mpe._ngrams[i - self.mpe.unigram_vocab_size]
            else:
                event_ids.append(i)

        performance = note_seq.Performance(
            quantized_sequence=None,
            steps_per_second=self.mpe._steps_per_second,
            num_velocity_bins=self.mpe._num_velocity_bins)

        for i in event_ids:
            # Ignore padding or stop at padding?
            if i - self.mpe.num_reserved_ids < 0:
                continue
            performance.append(self.mpe._encoding.decode_event(
                i - self.mpe.num_reserved_ids))

        ns = performance.to_sequence()
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        return pm
