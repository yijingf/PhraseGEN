import pretty_midi
import numpy as np
import scipy.io.wavfile


def main(midi_file, audio_file, fs=44100.0, to_int16=False):
    pm = pretty_midi.PrettyMIDI(midi_file)
    audio = pm.fluidsynth(fs=float(fs))

    if to_int16:
        audio *= 32768
        audio = audio.astype(np.int16)

    scipy.io.wavfile.write(audio_file, int(fs), audio)

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", dest="input", type=str,
                        help="Input MIDI file.")
    parser.add_argument("-o", "--output", dest="output", type=str,
                        help="Output WAV file")
    parser.add_argument("-f", "--fs", dest="fs", type=float, default=44100.0,
                        help="Sampling frequency. Defaults to 44.1kHz.")

    args = parser.parse_args()

    if not args.input:
        raise ValueError("Please specify input file name.")

    if not args.output:
        raise ValueError("Please specify output file name.")

    if args.output.split(".")[-1] != 'wav':
        raise ValueError("Incorrect output format.")

    main(args.input, args.output, args.fs)
