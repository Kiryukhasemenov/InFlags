#!/usr/bin/env python3

# USER INSTRUCTIONS:

# STEP 1: TRAINING THE DICTIONARY:
# python india-script.py -t -i path/to/training/text -o path/to/dictionary.json
# STEP 2: ENCODING THE FILE:
# python india-script.py -e -v path/to/dictionary.json -i path/to/input/text -o path/to/encoded/text
# STEP 3: DECODING THE FILE
# python india-script.py -d -v path/to/dictionary.json -i path/to/encoded/text -o path/to/decoded/text

import argparse

from india import InDia

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
    argparser.add_argument("-l", "--diacr_list", type=str, default="COMBINING ACUTE ACCENT,COMBINING CARON,COMBINING RING ABOVE", help="list of diacritics to be separated, defaults to Czech diacritics")
    args = argparser.parse_args()

    # if training: initialize bare, then run train with all inputs and arguments
    if args.train:
        india = InDia(pretrained_vocab=False)

        diacr_list = args.diacr_list.split(',')
        india.train_vocab(args.input, args.output, diacr_list=diacr_list, min_count=args.min_count, order_mode='freq')
        return 'vocabulary trained and saved to ' + args.output + ', for encoding and decoding refer to dict with --vocabulary or -v parameter.'

    elif args.encode:
        india = InDia(pretrained_vocab=True, vocab_file=args.vocabulary)
        india.encode(args.input, args.output)
        return 'encoded and saved to ' + args.output

    elif args.decode:
        india = InDia(pretrained_vocab=True, vocab_file=args.vocabulary)
        india.decode(args.input, args.output)
        return 'decoded and saved to ' + args.output

    else:
        return 'specify either --train or --encode or --decode'

if __name__ == "__main__":
    main()

