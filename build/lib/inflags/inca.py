import sys
import argparse
import unicodedata
import collections
import json
import re

class InCa:
    def __init__(self, pretrained_vocab=False, vocab_file=None):
        '''
        :param pretrained_vocab:
        :param vocab_file:
        '''
        self._ALPHANUMERIC_CHAR_SET = set(chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i))[0] in "LN")

        if pretrained_vocab:
            self.vocab, self.config = self._load_vocab(vocab_file)

        else:
            print('InCa initiated; please train vocab provided with necessary configurations (use train_vocab() method) or load vocab from file (use vocab_file argument for initialization).')

    def train_vocab(self, data_fname: str, vocab_fname: str, min_count=1, flags={'upper': 'ꔅ', 'title': 'ꔆ', 'lower': 'ꔪ', 'allcaps': 'ꔫ'}, include_allcaps=False, include_sent_initial=False):
        '''

        :param data_fname: str, path to the data file
        :param vocab_fname: str, path to the vocab file to be saved
        :param min_count: int, minimum frequency of a word to be included in the vocab
        :param flags: dict {casing: flag}, flags to be used for case marking
        :param include_allcaps: bool, whether to include all-caps words while building the vocab
        :param include_sent_initial: bool, whether to include sentence initial words while building the vocab
        TODO: :param flag_location: 'l' or 'r', whether to put flags before or after the word
        :return:
        '''
        
        # create configuration
        self.config = {'min_count': min_count, 'flags': flags, 'include_allcaps': include_allcaps, 'include_sent_initial': include_sent_initial}

        # count all possible casings for each base
        counts = self._collect_counts(data_fname)
        # save the most frequent ones
        self.vocab = self._save_vocab(counts)
        
        # Save self.config and vocab as a JSON file
        with open(vocab_fname, 'w') as vocab_file:
            json.dump({'config': self.config, 'vocab': self.vocab}, vocab_file, ensure_ascii=False, indent=4)

    def _tokenize(self, text):
        '''
        WORD tokenization function with the same principle as in tensor2tensor pre-tokenizer: split by spaces and non-alphanumeric characters.
        :param text: str, single line of text
        :return: list of tokens
        '''
        if not text:
            return []
        ret = []
        token_start = 0
        is_alnum = [c in self._ALPHANUMERIC_CHAR_SET for c in text]
        for pos in range(1, len(text)):
            if is_alnum[pos] != is_alnum[pos - 1]:  # what is that?
                token = text[token_start:pos]
                if token != " " or token_start == 0:
                    ret.append(token)
                token_start = pos
        final_token = text[token_start:]
        ret.append(final_token)
        return ret

    def _detokenize(self, tokens):
        """
        WORD detokenization function with the same principle as in tensor2tensor pre-tokenizer: join by spaces and non-alphanumeric characters.

        :param tokens: List, tokens to be detokenized.

        :return: A single string reconstructed from the list of tokens.

        """
        token_is_alnum = [t[0] in self._ALPHANUMERIC_CHAR_SET for t in tokens]
        ret = []
        for i, token in enumerate(tokens):
            if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
                ret.append(" ")
            ret.append(token)
        return "".join(ret)

    def _collect_counts(self, data_fname):
        """
        collects counts of all possible casings for each base

        :param data_fname: Path to the input data file.
        :type data_fname: str
        :return: dict containing bases (lowercase tokens) as keys and their respective
                 case-sensitive occurrence counts stored in a nested Counter.
        """
        counts = {}
        with open(data_fname) as fileobject:
            c = 0
            for line in fileobject:
                c += 1
                line = line.rstrip('\n\r')
                if c % 10000 == 0:
                    print(c)
                    print(type(counts), len(counts))

                # all-uppercase lines should not bias the statistics -> by default we exclude them
                if not self.config['include_allcaps']:
                    if line.isupper():
                        continue
                line = unicodedata.normalize("NFC", line)

                was_first_alphanum_token = False
                for token in self._tokenize(line):
                    # omitting the non-alphanumeric sequences
                    if token[0] not in self._ALPHANUMERIC_CHAR_SET:
                        continue
                    # if we are facing the sentence-initial word - we exclude it from statistics and go to the next token
                    if not self.config['include_sent_initial']:
                        if not was_first_alphanum_token:
                            was_first_alphanum_token = True
                            continue
                    else:
                        was_first_alphanum_token = True
                    base = token.lower()
                    is_cased = base.upper() != base
                    # if the word is case-able, we add it to its counter (initiate the Counter if necessary)
                    if is_cased:
                        if base not in counts:
                            counts[base] = collections.Counter()
                        counts[base][token] += 1

        return counts

    def _save_vocab(self, counts):
        """
        Saves the vocabulary as a flat dictionary {base: casing}. Works only for words which meet the conditions:
        frequency is above the minimum count, the most common form differs from the base.

        :param counts: output of the _collect_counts function, nested dictionary for all possible casing variants.
        :return: dict of the format {base: casing}
        """
        vocab = {}
        # iterating over all dictionary keys (lowercased words)
        for base in counts:
            # ordering the casing variants by their frequency
            form, count = counts[base].most_common(1)[0]
            # if the most frequent variant is not lowercased and if its count is higher than minimal: write it to vocab
            if count >= self.config['min_count'] and form != base:
                # print(form)
                vocab[base] = form
        return vocab


    def _load_vocab(self, vocab_fname):
        '''
        loads the vocabulary and configurations from a JSON file
        :param vocab_fname: str, path to the vocab file
        :return config: dictionary with configuration details, such as minimal counts, flags etc.
        :return vocab: dictionary with words and their most frequent casings
        '''
        with open(vocab_fname, 'r') as vocab_file:
            data = json.load(vocab_file)

        return data['vocab'], data['config']

    def encode(self, input_fname: str, output_fname: str, naive_encoding=False):
        """
        Encodes the content of an input file and writes the encoded output to another file.

        The method applies dictionary-based encoding, and writes the encoded content to the output file. 
        The "naive-encoding" option allows dictionary-free encoding (each cased word is explicitly flagged).

        :param input_fname: str, The file path to the input file containing raw data.
        :param output_fname: str, The destination file path where the encoded content will be written.
        :param naive_encoding: bool, A flag to specify whether to use naive encoding.
        :return: writes encoded data to the specified output file.
        """
        with open(input_fname, 'r') as input_fileobject, open(output_fname, 'w') as output_fileobject:
            for line in input_fileobject:
                encoded_line = self._encode_line(line, naive_encoding=naive_encoding)
                output_fileobject.write(encoded_line)
        return None

    def encode_string(self, string, naive_encoding=False):
        encoded_string = []
        for line in string.split('\n'):
            encoded_string.append(self._encode_line(line, naive_encoding=naive_encoding))
        return '\n'.join(encoded_string)

    def _encode_line(self, line, naive_encoding=False):
        '''
        line-by-line encoding function

        :param line: str, single line of text to be encoded.
        :param naive_encoding: bool, whether to use naive encoding (each cased word is explicitly flagged).
        :return: str, encoded line of text.
        '''
        
        line = unicodedata.normalize("NFC", line)
        if not self.config['include_allcaps']:
            if line.isupper() and " " in line:  # whole line is uppercased AND there is more than one word
                return f"{self.config['flags']['allcaps']} {line.lower()}"

        result = []

        was_first_alphanum_token = False

        for token in self._tokenize(line):
            if naive_encoding:
                result.append(self._encode_token_naive(token))
            else:
                if token[0] not in self._ALPHANUMERIC_CHAR_SET:
                    result.append(token)
                else:
                    if not self.config['include_sent_initial']:
                        if not was_first_alphanum_token:
                            result.append(self._encode_token(token, True))
                            was_first_alphanum_token = True  # for the beginning of sentence; `alphanumeric` - cause it may be starting with quotation marks
                        else:
                            result.append(self._encode_token(token, False))
                    else:
                        result.append(self._encode_token(token, False))
        return self._detokenize(result)

    def _encode_token(self, token, is_first=False):
        '''
        word-level encoding function

        :param token: str, single word to be encoded.
        :param is_first: bool, whether the word is in the beginning of the sentence.
        :return: str, either lowercased token or flag + lowercased token.
        '''
        # A mix of lowercase and uncased letters is OK
        lc_token = token.lower()
        vocab_form = self.vocab.get(lc_token)
        if vocab_form:  # if the lowercased token already in dict
            if token == vocab_form:
                return lc_token
            # else: the token is not the most frequent casing
            if is_first:  # if the word is in the beginning of the sentence AND not most frequent form
                if token == token.capitalize():
                    return f"{self.config['flags']['title']} {lc_token}"
            else:
                if token == lc_token:
                    #                print('gotten there?')
                    return f"{self.config['flags']['lower']} {lc_token}"
        if token == lc_token:
            if is_first:
                if token == token.capitalize():  # ? like Chinese chars?
                    return lc_token
                else:
                    #                print('somehow gotten here')
                    return f"{self.config['flags']['lower']} {lc_token}"  # if it is in the beginning BUT is not starting with capital - put LOWERCASE
            else:
                return lc_token
        if token.isupper():
            return f"{self.config['flags']['upper']} {lc_token}"  # single-char words like "I" fall into this category; we can implement favor-title just swap lines 76 and 78
        if token == token.capitalize():  # if token[0].isupper() and token[1:].islower():
            if is_first:  # this happens because when we decode we'll still put capitalization in the beginning of the sentence
                return lc_token
            else:
                return f"{self.config['flags']['title']} {lc_token}"
        return token

    def _encode_token_naive(self, token, mode='favor-upper'):
        '''
        naive encoding function - each word is explicitly flagged according to its casing.

        :param token: str
        :param mode: if 'favor-title': words like 'I' (1-char, uppercase) are considered titlecased
                     if 'favor-upper': considered uppercased
        :return: [optionally UPPERCASE/TITLECASE] token
        '''
        lc_token = token.lower()
        if token == lc_token:
            return token
        if len(token) == 1:
            if token.isupper():
                if mode == 'favor-title':
                    return f"{self.config['flags']['title']} {lc_token}"
                elif mode == 'favor-upper':
                    return f"{self.config['flags']['upper']} {lc_token}"
        else:
            if token.isupper():
                return f"{self.config['flags']['upper']} {lc_token}"
            if token.istitle():
                return f"{self.config['flags']['title']} {lc_token}"

        return token  # TODO: the mixed-case is saved as it is

    def decode(self, input_fname: str, output_fname: str, naive_decoding=False):
        """
        Decodes the content of an input file and writes the decoded content to an output file.
        The decoding can be toggled between naive and advanced methods based on the naive_decoding variable.

        :param input_fname: str, Name of the input file to be decoded
        :param output_fname: str, Name of the output file where the decoded content will be written
        :param naive_decoding: bool, whether naive decoding should be used.
        :return: writes decoded data to the specified output file.
        """

        with open(input_fname, 'r') as input_fileobject, open(output_fname, 'w') as output_fileobject:
            for line in input_fileobject:
                decoded_line = self._decode_line(line, naive_decoding=naive_decoding)
                output_fileobject.write(decoded_line)
        return None

    def decode_string(self, string, naive_decoding=False):
        decoded_string = []
        for line in string.split('\n'):
            decoded_string.append(self._decode_line(line, naive_decoding=naive_decoding))
        return '\n'.join(decoded_string)

    def _decode_line(self, line, naive_decoding=False):
        """
        line-level decoding function.

        :param line: str, The line of text to decode.
        :param naive_decoding: bool, If True, disables advanced vocabulary-based decoding mechanisms.
        :return: str, The decoded text line.

        """

        if not self.config['include_allcaps']:
            if line.startswith(f"{self.config['flags']['allcaps']} "):  # whole line is uppercased
                return unicodedata.normalize("NFC", line[2:].upper())
        result = []
        action, was_first_alphanum_token = None, False
        for token in self._tokenize(line):
            if token[0] not in self._ALPHANUMERIC_CHAR_SET:
                result.append(token)
            elif token in f"{self.config['flags']['upper']}{self.config['flags']['title']}{self.config['flags']['lower']}":
                action = token
            else:
                if action == self.config['flags']['upper']:
                    token = token.upper()
                elif action == self.config['flags']['title']:
                    token = token.capitalize()
                elif action == self.config['flags']['lower']:
                    token = token.lower()
                else:
                    if not naive_decoding:
                        vocab_form = self.vocab.get(token)
                        if vocab_form:
                            token = vocab_form
                        elif not self.config['include_sent_initial'] and not was_first_alphanum_token and token == token.lower():
                            token = token.capitalize()
                result.append(token)
                action, was_first_alphanum_token = None, True
        return unicodedata.normalize("NFC", self._detokenize(result))

