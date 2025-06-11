import os
from io import open as open_io
import torch

from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), train=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path, train=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        if train:
            words = []
            # Add words to the dictionary
            with open_io(path, 'r', encoding="utf8") as f:
                for line in f.read().lstrip('. ').rstrip(' .').split(' . '):
                    words += line.split() + ['<eos>']
            self.dictionary.add_word('<unk>')
            dictionary = Counter(words)
            for word in dictionary:
                if dictionary[word] > 1:
                    self.dictionary.add_word(word)
            

        # Tokenize file content
        with open_io(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f.read().lstrip('. ').rstrip(' .').split(' . '):
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    try:
                        ids.append(self.dictionary.word2idx[word])
                    except KeyError:
                        ids.append(self.dictionary.word2idx['<unk>'])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
