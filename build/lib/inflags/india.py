#!/usr/bin/env python3
# coding=utf-8

import sys
import unicodedata
import collections
import json

class InDia:
    def __init__(self, pretrained_vocab=False, vocab_file=None):
        '''
        initializes the InDia class, imports vocabulary and configurations if provided.

        :param pretrained_vocab: bool, whether to load a pretrained vocab from file.
        :param vocab_file: path to the vocab file, if pretrained_vocab is True.
        '''
        if pretrained_vocab:
            self.vocab, self.config = self._load_vocab(vocab_file)
            self._ALPHANUMERIC_CHAR_SET = self._create_alphanumeric_char_set()[1]
        else:
            print('InDia initiated; please train vocab provided with necessary configurations (use train_vocab() method) or load vocab from file (use vocab_file argument for initialization).')

    def train_vocab(self, data_fname, vocab_fname, diacr_list=['COMBINING ACUTE ACCENT', 'COMBINING CARON', 'COMBINING RING ABOVE'], min_count=1, order_mode='freq'):
        """
        Train a vocabulary based on input text, saving character frequencies and configurations.

        :param data_fname: str, The path to the training data file.
        :param vocab_fname: str, path to the JSON vocab file to be saved
        :param diacr_list: list[str] A list of diacritic marks to consider during processing.
                            The names should follow the Unicode standard (use `unicodedata.name(char)` to get the name).
                            The default value includes the 3 standard Czech diacritics.
        :param min_count: int, minimum frequency of a word to be included in the vocab
        :param order_mode: str, {'freq' or 'base'}: defines which base form is marked in the dictionary (most frequent or the un-diacritized one).
        :return: writes the vocab and configurations to the vocab_fname.
        """
        self._ALPHANUMERIC_CHAR_SET, config_dict = self._create_flag_set(diacr_list)
        config_dict['MIN_COUNT'] = min_count
        config_dict['ORDER_MODE'] = order_mode

        self.config = config_dict

        # count all possible casings for each base
        counts = self._collect_counts(data_fname)
        # save the most frequent ones
        self.vocab = self._save_vocab(counts, min_count)

        # Save self.config and vocab as a JSON file
        with open(vocab_fname, 'w') as vocab_file:
            json.dump({'config': self.config, 'vocab': self.vocab}, vocab_file, ensure_ascii=False, indent=4)


    def _load_vocab(self, vocab_file):
        '''
        loads the vocabulary and configurations from a JSON file
        :param vocab_fname: str, path to the vocab file
        :return config: dictionary with configuration details, such as minimal counts, flags etc.
        :return vocab: dictionary with words and their most frequent casings
        '''
        with open(vocab_file, 'r') as f:
            data = json.load(f)

        return data['vocab'], data['config']

    def _create_alphanumeric_char_set(self):
        '''
        necessary for tokenization and detokenization, separates the characters into alphanumeric and non-alphanumeric (incl. flags)
        :return CHAR FLAGS: list of characters that are considered as flags
        :return _ALPHANUMERIC_CHAR_SET: set of characters that are considered as alphanumeric
        '''
        # TODO: make it adjustable depending on variety of characters
        CHAR_RANGE = (0x24B6, 0x24EA)  # Circled Latin A-Za-z
        CHAR_FLAGS = [chr(c) for c in range(CHAR_RANGE[0], CHAR_RANGE[1] + 1)]
        _ALPHANUMERIC_CHAR_SET = set(chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i))[0] in "LN") | set(CHAR_FLAGS)
        return CHAR_FLAGS, _ALPHANUMERIC_CHAR_SET

    def _create_flag_set(self, diacr_list):
        """
        Creates a flag set for diacritization symbols, maps them to specific flags, stores the mappings in a dictionary,.

        :param diacr_list: List of names of diacritization symbols to be processed.

        :return: _ALPHANUMERIC_CHAR_SET: set of characters that are considered as alphanumeric.
        :return: A dictionary containing the following keys:
                 - 'UPPER_LEVEL': A dictionary with `KEY_FLAG` (splitting the ids of characters (int) that have diacritization flags)
                                  and `DICT_FLAG` (showing the beginning of the diacritization flag sequence and splitting the key sequence and value sequence)
                 - 'NAME2FLAG': A dictionary mapping diacritization names to their corresponding flags,
                                including special symbols like 'OOV' and 'BARE'.
                 - 'FLAG2NAME': A dictionary mapping flags back to their corresponding diacritization names.
        """

        CHAR_FLAGS, _ALPHANUMERIC_CHAR_SET = self._create_alphanumeric_char_set()
        KEY_FLAG = 'ꕐ'  # lower-lever flag splitting the keys
        DICT_FLAG = 'ꕑ'  # upper-level flag splitting key seq and value seq
        upper_level = {'KEY_FLAG': KEY_FLAG, 'DICT_FLAG': DICT_FLAG}
        BARE_FLAG = '⓿' # flag showing no diacritization for character
        OOV_FLAG = '⓪'  # U+24EA

        assert len(diacr_list) <= len(CHAR_FLAGS), 'the range of diacritization symbols is larger than flag charset; choose another range'
        name2flag = {}
        for name, flag in zip(diacr_list, CHAR_FLAGS):
            name2flag[name] = flag

        name2flag['OOV'] = OOV_FLAG
        name2flag['BARE'] = BARE_FLAG
        flag2name = {v: k for k, v in name2flag.items()}

        return _ALPHANUMERIC_CHAR_SET, {'UPPER_LEVEL': upper_level, 'NAME2FLAG': name2flag, 'FLAG2NAME': flag2name}

    def _collect_counts(self, input_file):
        """
        collects counts of all possible diacritization for each base

        :param data_fname: Path to the input data file.
        :type data_fname: str
        :return: dict containing bases (de-diacritized tokens) as keys and their respective
                 diacritization counts stored in a nested Counter.
        """
        counts = {}
        with open(input_file) as fileobject:
            for line in fileobject:
                line = line.rstrip('\n\r')
                for token in self._tokenize(line):
                    # strip diacritics
                    # add to counter the specific form
                    if token[0] not in self._ALPHANUMERIC_CHAR_SET:
                        continue
                    base_token = self._dediacritize(token)
                    if base_token not in counts:
                        counts[base_token] = collections.Counter()
                    counts[base_token][token] += 1

        return counts

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

    def _dediacritize(self, s):
        '''
        strips diacritics ONLY from the set of diacritics that are pre-defined in the config file
        :param s: str, single token
        :return: str, de-diacritized token
        '''
        stripped_line = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.name(c) not in set(self.config['NAME2FLAG'].keys()))
        return unicodedata.normalize('NFC', stripped_line)

    def _save_vocab(self, counts, min_count):
        '''
        writing the lines to dict
        the data is organized as a json dict of the format:
            {base: most_frequent_diacritization}
        , where base = dediacritized token
        :param counts: dict, output of _collect_counts()
        :param min_count: int, minimal frequency of a word to be included in the vocab
        :return: dict, vocab in the format {base: most_frequent_diacritization}
        '''
        json_line = {}
        for base_token in counts:
            # ordering the casing variants by their frequency
            form, count = counts[base_token].most_common(1)[0]
            # if the most frequent variant is not base and if its count is higher than minimal: write it to vocab
            if form != base_token and count >= min_count:
                json_line[base_token] = form

        return json_line

    def encode(self, input_fname: str, output_fname: str):
        """
        Encodes the content of an input file and writes the encoded output to another file.

        :param input_fname: str, The file path to the input file containing raw data.
        :param output_fname: str, The destination file path where the encoded content will be written.
        :return: writes encoded data to the specified output file.
        """
        with open(input_fname, 'r') as input_fileobject, open(output_fname, 'w') as output_fileobject:
            for line in input_fileobject:
                encoded_line = self._encode_line(line)
                output_fileobject.write(encoded_line)
        return None

    def encode_string(self, string):
        '''
        Encodes a string and returns it as a string. Analogous to encode() method.

        :param string: str
        :return: str
        '''
        encoded_string = []
        for line in string.split('\n'):
            encoded_string.append(self._encode_line(line))
        return '\n'.join(encoded_string)

    def _encode_line(self, line):
        """
        Encodes a line by de-diacritizing and putting tokens.

        :param line: str, input line of text to be encoded.
        :return: str, encoded version of the input line.
        """
        # TODO: in the other modes, implement encoding of the non-diacritized full line
        result = []
        for token in self._tokenize(line):
            if token[0] in self._ALPHANUMERIC_CHAR_SET:
                base = self._dediacritize(token)
                mostfreq_form = self.vocab.get(base, '')

                if mostfreq_form == token:
                    result.append(base)
                else:
                    result.append(self._encode_token(token, mostfreq_form))
            else:
                result.append(token)
        return self._detokenize(result)

    def _encode_token(self, input_token, mostfreq_token):
        """
        Encodes the diacritic differences between input and the most frequent diacritic word.

        takes on two words - the most frequent diacritization and the input word
        checks if it's the same diacritization
        if yes -> return base
        else -> return only the flags that are DIFFERENT from the most frequent diacritization

        :param input_token: str, The word for which the diacritic encoding is to be performed.
        :param mostfreq_token: str, most frequent diacritization of the base to compare against the input token.
        :return: str, (flags if necessary) and de-diacritized word.
        """

        input_diacr = self._detect_diacr(input_token)
        mostfreq_diacr = self._detect_diacr(mostfreq_token)
        encode_diacr = {}
        all_keys = set(input_diacr.keys()) | set(mostfreq_diacr.keys())
        for k in all_keys:
            if k in input_diacr.keys() and k in mostfreq_diacr.keys():
                if input_diacr[k] == mostfreq_diacr[k]:
                    pass
                else:
                    encode_diacr[k] = input_diacr[k]
            elif k in input_diacr.keys() and k not in mostfreq_diacr.keys():
                encode_diacr[k] = input_diacr[k]
            elif k not in input_diacr.keys() and k in mostfreq_diacr.keys():
                encode_diacr[k] = self.config['NAME2FLAG']['BARE']
        if encode_diacr == {}:
            return self._dediacritize(input_token)

        encoded_flags = self._flag_processing(encode_diacr, mode='encode')
        return f'{encoded_flags} {self._dediacritize(input_token)}'

    def decode(self, input_fname: str, output_fname: str):
        """
        Decodes the content of an input file and writes the decoded content to an output file.
        The decoding can be toggled between naive and advanced methods based on the naive_decoding variable.

        :param input_fname: str, Name of the input file to be decoded
        :param output_fname: str, Name of the output file where the decoded content will be written
        :return: writes decoded data to the specified output file.
        """

        with open(input_fname, 'r') as input_fileobject, open(output_fname, 'w') as output_fileobject:
            for line in input_fileobject:
                decoded_line = self._decode_line(line)
                output_fileobject.write(decoded_line)
        return None

    def decode_string(self, string):
        '''
        decodes a string and returns it as a string. Analogous to decode() method.

        :param string: str
        :param naive_decoding: bool
        :return: str
        '''
        decoded_string = []
        for line in string.split('\n'):
            decoded_string.append(self._decode_line(line))
        return '\n'.join(decoded_string)

    def _decode_line(self, line):  # , writing_mode='base'
        '''
        decoding function, inspired from inca
        if token is non-alphanumeric - leave intact
        if token is in set of flags (including unseen) - assign diacr_id
        if token is word after the flag - find it in the dictionary (or if UNSEEN FLAG - leave intact)
        '''
        # TODO: in the other modes, implement decoding of the non-diacritized full line
        result = []
        diacr_id = {}
        for token in self._tokenize(line):  # TODO: check all
            if token[0] not in self._ALPHANUMERIC_CHAR_SET:
                result.append(token)
            elif token.startswith(self.config['UPPER_LEVEL']['DICT_FLAG']):  # , UNSEEN_FLAG TODO: generalize all flags into one group?
                diacr_id = token
            else:
                # base_token = dediacritize(token) # TODO: delete for speedup
                mostfreq_form = self.vocab.get(token, '')
                decoded_token = self._decode_token(token, diacr_id, mostfreq_form)
                result.append(decoded_token)
                diacr_id = {}
        return self._detokenize(result)

    def _decode_token(self, input_base, diacritization, mostfreq_token):
        """
        Decodes a base by combining explicit diacritization flags with the most frequent token's diacritization.
        The priority is given to the explicit flags.

        1. lookup the token in vocabulary
        2. if it is - take it; else - ''
        3. make voc_diacritization for vocab token
        3. make union of two diacritizations keys
        4. for key in union:
            if key in curr_diacritization: write it down
            elif key in voc_diacritization and not in curr_diacritization: write voc_diacritization[key]

        :param input_base: str, The base input token to be decoded.
        :param diacritization: dict[char_id: diacritization flag], A dictionary containing explicit flags for the input_base token.
        :param mostfreq_token: str, The most frequent diacritization associated with input_base,
        :return: str, The decoded token reconstructed with combined diacritization information from input_base and mostfreq_token.
        """

        if diacritization != {}:
            diacritization = self._flag_processing(diacritization, mode='decode')
        mostfreq_diacr = self._detect_diacr(mostfreq_token)

        all_keys = set(diacritization.keys()) | set(mostfreq_diacr.keys())
        decode_dict = {}
        for k in all_keys:
            if k in diacritization:
                if diacritization[k] != self.config['NAME2FLAG']['BARE']:
                    decode_dict[k] = diacritization[k]
            elif k not in diacritization and k in mostfreq_diacr.keys():
                decode_dict[k] = mostfreq_diacr[k]

        decoded_token = self._restore_diacr(input_base, decode_dict)
        return decoded_token

    def _detect_diacr(self, w):
        """
        Detects diacritics in a given string and maps their positions and types.

        1. iterates over characters
        2. if a character is splittable and one of the characters is a diacritic (in the list defined in config):
             returns the id of this character and the type of diacritic

        :param w: str, The input word to process.
        :return: dict[int, flag] A dictionary where keys are the positions of the detected diacritics in
            the input string and values are their corresponding types from the configuration.
        """

        k_v = {}
        for idx, char in enumerate(w):
            split_char = unicodedata.normalize('NFD', char)
            if len(split_char) > len(char):
                for s in split_char:
                    if unicodedata.name(s) in self.config['NAME2FLAG'].keys():  # unicodedata.category(s) == 'Mn':
                        k_v[idx] = self.config['NAME2FLAG'][unicodedata.name(s)]
                    else:
                        pass

        return k_v

    def _restore_diacr(self, w, diacr_dict):
        """
        Restores diacritical marks in a given word based on the provided mapping.
        Applies diacritics at specified positions, utilizing Unicode NFC notrmaliztion to form valid characters.

        :param w: str, The word to which diacritical marks need to be restored.
        :param diacr_dict: dict[int, str], mapping character indices in the word to
            diacritical mark identifiers.
        :return: str, The diacritized word.
        """
        final_w = w
        for diacr_id in diacr_dict.keys():
            diacr_name = self.config['FLAG2NAME'][diacr_dict[diacr_id]]
            if diacr_name != 'OOV':
                diacritics = unicodedata.lookup(diacr_name)
                try:
                    combination = final_w[diacr_id] + diacritics
                    new_char = unicodedata.normalize('NFC', combination)
                    final_w = final_w[:diacr_id] + new_char + final_w[diacr_id + 1:]
                except:
                    pass
            else:
                pass
        return final_w

    def _flag_processing(self, input_data, mode='encode'):
        """
        Transforms a flag dictionary {char_id: flag} into a string format (for encoding) or vice versa.

        if mode == 'encode': takes a dictionary and transforms it into a string of the format:
           DICT_FLAG k1 KEY_FLAG k2 KEY_FLAG ... KEY_FLAG kn DICT_FLAG v1v2...vn, where
            - DICT_FLAG is the flag showing the beginning of a diacritization encoding and border between keys and values,
            - KEY_FLAG is the flag splitting the key IDs between each other.

        if mode == 'decode': takes a string and transforms it into a dictionary of the format: {char_id: flag}

        :param input_data: dict when in 'encode' mode, str when in 'decode' mode.
        :param mode: str, either 'encode' or 'decode'.
        :return: The processed result. str when mode='encode', dict when mode='decode'
        """
        if mode == 'encode':
            keys = sorted(input_data.keys())
            values = ''.join([input_data[k] for k in keys])

            encoded_keys = self.config['UPPER_LEVEL']['KEY_FLAG'].join([str(k) for k in keys])
            encoded_line = self.config['UPPER_LEVEL']['DICT_FLAG'] + encoded_keys + self.config['UPPER_LEVEL']['DICT_FLAG'] + values
            return encoded_line

        elif mode == 'decode':
            enc_keys, enc_values = input_data.split(self.config['UPPER_LEVEL']['DICT_FLAG'])[1:]
            keys = [int(k) for k in enc_keys.split(self.config['UPPER_LEVEL']['KEY_FLAG'])]
            decoded_dict = {pair[0]: pair[1] for pair in zip(keys, list(enc_values))}
            return decoded_dict

