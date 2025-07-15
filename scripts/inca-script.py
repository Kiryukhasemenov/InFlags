#!/usr/bin/env python3

# USER INSTRUCTIONS:

# STEP 1: TRAINING THE DICTIONARY:
# python inca-script.py -t -i path/to/training/text -o path/to/dictionary.json
# STEP 2: ENCODING THE FILE:
# python inca-script.py -e -v path/to/dictionary.json -i path/to/input/text -o path/to/encoded/text
# STEP 3: DECODING THE FILE
# python inca-script.py -d -v path/to/dictionary.json -i path/to/encoded/text -o path/to/decoded/text

import argparse

from inca import InCa

def main():
    argparser = argparse.ArgumentParser()
    # file paths:
    argparser.add_argument("-i", "--input", type=str, help="path to input file")
    argparser.add_argument("-o", "--output", type=str, help="path to output file")
    argparser.add_argument("-v", "--vocabulary", help="a vocabulary to be used")

    # three functions:
    argparser.add_argument("-t", "--train", action='store_true', default=False, help="train a vocabulary")
    argparser.add_argument("-e", "--encode", action='store_true', default=False, help="encoding of the input file")
    argparser.add_argument("-d", "--decode", action='store_true', default=False, help="run a decoder (encoder is the default)")

    # other parameters:
    argparser.add_argument("-m", "--min_count", type=int, default=1, help="vocabulary items must be at least so frequent")
    argparser.add_argument("-n", "--naive", action='store_true', default=False, help="naive preprocessing algorithm")
    argparser.add_argument("-a", "--include_allcaps", action='store_true', default=False, help="including allcaps into statistics")
    argparser.add_argument("-isi", "--include_sent_initial", action='store_true', default=False, help="include sentence-intial tokens into statistics and put flags to them")
    argparser.add_argument("-f", "--flags", type=str, default='upper:ꔅ,lower:ꔪ,title:ꔆ,allcaps:ꔫ', help="dictionary of flags, pass in the form of key:value,key:value")
    args = argparser.parse_args()

    # if training: initialize bare, then run train with all inputs and arguments
    if args.train:
        inca = InCa(pretrained_vocab=False)

        flags = {}
        for pair in args.flags.split(','):
            key, value = pair.split(':')
            flags[key] = value
        inca.train_vocab(args.input, args.output, min_count=args.min_count, include_allcaps=args.include_allcaps, include_sent_initial=args.include_sent_initial, flags=flags)
        return 'vocabulary trained and saved to ' + args.output + ', for encoding and decoding refer to dict with --vocabulary or -v parameter.'

    elif args.encode:
        inca = InCa(pretrained_vocab=True, vocab_file=args.vocabulary)
        inca.encode(args.input, args.output, naive_encoding=args.naive)
        return 'encoded and saved to ' + args.output

    elif args.decode:
        inca = InCa(pretrained_vocab=True, vocab_file=args.vocabulary)
        inca.decode(args.input, args.output, naive_decoding=args.naive)
        return 'decoded and saved to ' + args.output

    else:
        return 'specify either --train or --encode or --decode'

if __name__ == "__main__":
    main()

