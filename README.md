# InFlags
Python package for dictionary-based inline tokenization preprocessing (featuring **In**line **Flags**).

## Overview

InCa is a system that is used **prior** to training your subword tokenizer in order to make encoding of differently cased words consistent. This is done through inline **casing approach**: the information about the casing of a word is re-allocated to the left of the word with a specific **flag**. The special feature of our method is that we only allocate the flags to the casings of the words which are **not the most frequent ones**. This is done through keeping the information about the most frequent casing for each word in a pre-trained **dictionary**. 

Similarly to InCa, this is an **inline diacritization** approach based on dictionary. Firstly, we collect information about which diacritization is most frequent for each **base** (unique non-diacritized character sequence), and then at encoding we only mark the characters that are differing from the most frequent diacritization or are out of dictionary. 

**An important difference:** while in InCa, the flags could be stored as single characters, here they are multi-character because we need to deterministically show which character IDs have which diacritics. The syntax of a flag is as follows: 

```DICT_FLAG k1 KEY_FLAG k2 ... KEY_FLAG kn DICT_FLAG v1 v2 ... vn```. For example: 
```ꕑ3ꕐ4ꕑⒷⒶ konci```.

where:
 - `ꕑ` is a `DICT_FLAG` used to indicate two things:
   - the beginning of the sequence of diacritization-related flags (first occurrence)
   - the border between the key sequence and value sequence (second occurrence)
 - `ꕐ` is a `KEY_FLAG` separating the character IDs (because they are encoded in regular numbers)
 - `Ⓑ` and `Ⓐ` are flags showing that the characters with IDs 3 and 4, respectively, should be diacritized in a particular way
 - `konci` is a word to be diacritized


For more details, please refer to:
1. [paper](https://openreview.net/pdf?id=9GwVWxjVmN)  presented at [Tokenization Workshop](https://tokenization-workshop.github.io) @ICML2025, 
2. [video demonstration](https://www.youtube.com/watch?v=XgDPXWsQEwI),
3. [summarizing poster](InCa_InDia_poster.pdf)


## Installation

Currently, only Github installation is available; the PyPI package will be accessible soon.

For now, just run: 

```aiignore
python3 -m pip install git+https://github.com/Kiryukhasemenov/InFlags.git
```

## InCa - Inline Casing

[Python API](#quick-start---inca-api) ||| [Command line tool](#inca-command-line-tool)

### Quick Start - InCa API:

**0. Import package**

```aiignore
from inflags.inca import InCa
```

#### 1. Train or import dictionary

First, you need to have the dictionary where the most frequent casing of each word is stored. 

If you don't have a dictionary yet, you can train it with the following code: 

```aiignore
inca = InCa(pretrained_dictionary=False)

inca.train_dictionary('path/to/training/data', 'path/to/dictionary.json')
```

If you already have a pre-trained dictionary, you can import it at the instantiation of the InCa class: 

```aiignore
inca = InCa(pretrained_dictionary=True, dictionary_file='path/to/dictionary.json')
```

##### Additional parameters: 

- `min_count`: str, defaults to 1. Minimal number of occurrences of a particular casing to be included into the dictionary.
- `flags`: dict[str: str], defaults to {'upper': 'ꔅ', 'title': 'ꔆ', 'lower': 'ꔪ', 'allcaps': 'ꔫ'}. The inventory of flags to use. In default implementation, all flags are single-character (this ensures that they are atomic in any subword tokenizer). 
- `include_allcaps`: bool, defaults to False. Whether to include all-caps sentences in counts while building the dictionary.
- `include_sent_initial`: bool, defaults to False. Whether to include sentence initial words while building the dictionary.

#### 2. Encode text

When the dictionary is trained, you can apply it to your input. 

If you have a long document to be encoded, use the `encode` method: 

```aiignore
inca.encode('path/to/input/file', 'path/to/output/file')
```

If you have shorter sequences and want to process them later in the code, use the `encode_string` method:

```aiignore
inca.encode_string('Encode this SHORT string in English.')
```

If you trained the dictionary on English data, then "English" would probably be in the dictionary with title-cased spelling as most frequent; and "SHORT" would not. So most probably the output would be as follows:

```aiignore
> 'encode this ꔅ short string in english.'
```

, where `ꔅ` is an upper-case flag. The word "Encode" is not marked since it's sentence-initial, the word "English" is not marked since it is its most frequent spelling.

##### Additional paramters:
- `naive_encoding`: bool, defaults to False. Encoding the string or document ignoring the dictionary, i.e. putting explicit flags wherever the word is not lower-cased. 
#### 3. Decode

The algorithm is reversible, so you can decode such a string to a traditional cased sequence.

For files, use the following command: 

```aiignore
inca.decode('path/to/encoded/file', 'path/to/decoded/file')
```

For string variables in the code, use: 
```aiignore
inca.decode_string('encode this ꔅ short string in english.')
> 'Encode this SHORT string in English.'
```

##### Additional paramters:
- `naive_decoding`: bool, defaults to False. Decoding the string or document ignoring the dictionary, i.e. putting explicit flags wherever the word is not lower-cased. 

### InCa: Command Line Tool

There is a wrapper for all commands described above, you can find them in the [scripts](scripts/inca-script.py) file.

```aiignore
# 1. Train the dictionary:
python inca-script.py -t -i path/to/training/text -o path/to/dictionary.json
# 2. Encode a file:
python inca-script.py -e -dict path/to/dictionary.json -i path/to/input/text -o path/to/encoded/text
# 3. Decode a file:
python inca-script.py -d -dict path/to/dictionary.json -i path/to/encoded/text -o path/to/decoded/text


```

#### TODO:

- [ ] add unit tests 
- [ ] publish code on PyPI
- [ ] add naive encoding/decoding without initializing the dictionary
- [ ] handle OOV mixed-cased words in a more consistent manner
- [ ] handle the Turkish letters "İ"/"i" VS "I"/"ı" (at the moment this is not fully reversible)
- [ ] test the multi-character flags (for the compatibility with the Sentencepiece's [user-defined symbols](https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md))


## InDia - Inline Diacritization

[Python API](#quick-start---india-api) ||| [Command line tool](#india-command-line-tool)

### Quick Start - InDia API:

**0. Import package**

```aiignore
from inflags.india import InDia
```

#### 1. Train or import dictionary

**ATTENTION:** Contrary to InCa, this package is quite language-specific for Czech. Therefore, if you are using it for other language, you should first define what variety of the diacritics you will be working with. This will be explicitly fed as `diacr_list` parameter (see in [additional parameters below](#additional-parameters--1))

First, you need to have the dictionary where the most frequent casing of each word is stored. 

If you don't have a dictionary yet, you can train it with the following code: 

```aiignore
india = InDia(pretrained_dictionary=False)

india.train_dictionary('path/to/training/data', 'path/to/dictionary.json')
```

If you already have a pre-trained dictionary, you can import it at the instantiation of the InCa class: 

```aiignore
india = InDia(pretrained_dictionary=True, dictionary_file='path/to/dictionary.json')
```

##### Additional parameters: 

- `min_count`: str, defaults to 1. Minimal number of occurrences of a particular casing to be included into the dictionary.
- `diacr_list`: list[str], defaults to Czech diacritics ['COMBINING ACUTE ACCENT', 'COMBINING CARON', 'COMBINING RING ABOVE']. The inventory of diacritics that would be stripped away and flagged (the others will be kept untouched). If you are working with the language other than Czech, first define the list of language-specific diacritics. You can do it with `unicodedata.name(char)` to get the name of each diacritic sign. 
#### 2. Encode text

If you have a long document to be encoded, use the `encode` method: 

```aiignore
india.encode('path/to/input/file', 'path/to/output/file')
```

If you have shorter sequences and want to process them later in the code, use the `encode_string` method:

```aiignore
india.encode_string('Vyrobil více než 1 000 známek pro Švédsko a 28 dalších zemí.')
```

If you trained the dictionary on Czech FLORES data, then "Švédsko" would be in the dictionary with such a diacritization; and "zemí" would not. So the output would be as follows:

```aiignore
> 'Vyrobil vice nez 1 000 znamek pro Svedsko a 28 dalsich ꕑ3ꕑⒶ zemi.'
```

, where `ꕑ3ꕑⒶ` is a flag saying that character with ID 3 should be assigned with the diacritic "čárka".

#### 3. Decode

The algorithm is reversible, so you can decode such a string to a traditional cased sequence.

For files, use the following command: 

```aiignore
india.decode('path/to/encoded/file', 'path/to/decoded/file')
```

For string variables in the code, use: 
```aiignore
india.decode_string('Vyrobil vice nez 1 000 znamek pro Svedsko a 28 dalsich ꕑ3ꕑⒶ zemi.')
> 'Vyrobil více než 1 000 známek pro Švédsko a 28 dalších zemí.'
```


### InDia: Command Line Tool 

There is a wrapper for all commands described above, you can find them in the [scripts](scripts/india-script.py) file.

```aiignore
# 1. Train the dictionary:
python india-script.py -t -i path/to/training/text -o path/to/dictionary.json
# 2. Encode a file:
python india-script.py -e -dict path/to/dictionary.json -i path/to/input/text -o path/to/encoded/text
# 3. Decode a file:
python india-script.py -d -dict path/to/dictionary.json -i path/to/encoded/text -o path/to/decoded/text


```

#### TODO:

- [ ] add unit tests 
- [ ] refactor code for speed
- [ ] add naive encoding/decoding option without setting min_count to 10000000000
- [ ] introduce the word- or even sentence-level flags (e.g. "fully de-diacritized word")