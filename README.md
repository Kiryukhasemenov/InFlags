# InFlags
Python package for dictionary-based inline tokenization preprocessing

## Installation

Currently, only Github installation is available; the PyPI package will be accessible soon.

For now, just run: 

```aiignore
python3 -m pip install git+https://github.com/Kiryukhasemenov/InFlags.git
```

## InCa - Inline Casing

### Short description

You can use this system **prior** to training your subword tokenizer in order to make encoding of differently cased words consistent. This is done through inline **casing approach**: the information about the casing of a word is re-allocated to the left of the word with a specific **flag**. The special feature of our method is that we only allocate the flags to the casings of the words which are **not the most frequent ones**. This is done through keeping the information about the most frequent casing for each word in a pre-trained **dictionary** (aka **vocabulary** in code). 

For more details, please refer to the [paper presented at Tokenization Workshop @ICML2025](https://openreview.net/pdf?id=9GwVWxjVmN), or to its [video demonstration](https://www.youtube.com/watch?v=XgDPXWsQEwI). 

### Quick Start - API:

**0. Import package**

```aiignore
from inflags.inca import InCa
```

#### 1. Train or import vocabulary

First, you need to have the vocabulary where the most frequent casing of each word is stored. 

If you don't have a vocabulary yet, you can train it with the following code: 

```aiignore
inca = InCa(pretrained_vocab=False)

inca.train_vocab('path/to/training/data', 'path/to/vocab.json')
```

If you already have a pre-trained vocabulary, you can import it at the instantiation of the InCa class: 

```aiignore
inca = InCa(pretrained_vocab=True, vocab_file='path/to/vocab.json')
```

##### Additional parameters: 

- `min_count`: str, defaults to 1. Minimal number of occurrences of a particular casing to be included into the dictionary.
- `flags`: dict[str: str], defaults to {'upper': 'ꔅ', 'title': 'ꔆ', 'lower': 'ꔪ', 'allcaps': 'ꔫ'}. The inventory of flags to use. In default implementation, all flags are single-character (this ensures that they are atomic in any subword tokenizer). 
- `include_allcaps`: bool, defaults to False. Whether to include all-caps sentences in counts while building the vocab.
- `include_sent_initial`: bool, defaults to False. Whether to include sentence initial words while building the vocab.

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

If you trained the vocabulary on English data, then "English" would probably be in the dictionary with title-cased spelling as most frequent; and "SHORT" would not. So most probably the output would be as follows:

```aiignore
> 'encode this <U> short string in english.'
```

, where `<U>` is an upper-case flag. The word "Encode" is not marked since it's sentence-initial, the word "English" is not marked since it is its most frequent spelling.

##### Additional paramters:
- `naive_encoding`: bool, defaults to False. Encoding the string or document ignoring the vocabulary, i.e. putting explicit flags wherever the word is not lower-cased. 
#### 3. Decode

The algorithm is reversible, so you can decode such a string to a traditional cased sequence.

For files, use the following command: 

```aiignore
inca.decode('path/to/encoded/file', 'path/to/decoded/file')
```

For string variables in the code, use: 
```aiignore
inca_cs.decode_string('encode this <U> short string in english.')
> 'Encode this SHORT string in English.'
```

##### Additional paramters:
- `naive_decoding`: bool, defaults to False. Decoding the string or document ignoring the vocabulary, i.e. putting explicit flags wherever the word is not lower-cased. 

### Quick Start: Command Line Tool

There is a wrapper for all commands described above, you can find them in the [scripts](scripts/inca-script.py) file.

```aiignore
# 1. Train the vocabulary:
python inca-script.py -t -i path/to/training/text -o path/to/dictionary.json
# 2. Encode a file:
python inca-script.py -e -v path/to/dictionary.json -i path/to/input/text -o path/to/encoded/text
# 3. Decode a file:
python inca-script.py -d -v path/to/dictionary.json -i path/to/encoded/text -o path/to/decoded/text


```

## InDia - Inline Diacritization