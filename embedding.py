import sys
from typing import Union, Iterable

import numpy as np
from gensim.models import KeyedVectors, Word2Vec

# Our tokenizer is defined in this file because this
# class needs the tokenizer and our model/training
# files already import this file. We don't want to
# end up using inconsistent tokenizers between
# our files.

# Feel free to change the tokenizer. Note: (julian)
# was unable to get spacy to work. nltk seems to be
# mostly equivalent.

# [NLTK]
from nltk import word_tokenize
tokenize = word_tokenize

# [SpaCy]
# import spacy
# spacy_eng = spacy.load("en")
# tokenize = spacy_eng.tokenizer


def loadPretrained(filePath):
    """Opens a `.vec` file and returns word2index, index2word,
    and the matrix of the word embedding vectors"""

    wordVectors = KeyedVectors.load_word2vec_format(filePath)

    # Extract relevant data
    vectorSize = wordVectors.vector_size
    matrix = wordVectors.vectors
    index2word = wordVectors.index2word
    word2index = {word: index for index, word in enumerate(index2word)}

    def appendWordVector(matrix, word, vector):
        if word not in word2index:
            word2index[word] = len(matrix)
            matrix = np.append(matrix, [vector], axis=0)

        return matrix

    # This assigns [0, 0, ..., 0] to SOS and EOS if not in the pretrained data.
    matrix = appendWordVector(matrix, Token.SOS, np.zeros(vectorSize))
    matrix = appendWordVector(matrix, Token.EOS, np.zeros(vectorSize))

    # Padding is a vector of all zeros
    matrix = appendWordVector(matrix, Token.PAD, np.zeros(vectorSize))

    # Unknown gets a vector with random values
    matrix = appendWordVector(matrix, Token.UNK, np.random.rand(vectorSize))

    return word2index, index2word, matrix


# Enumeration
class Token():
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"
    PAD = "<pad>"


class Vocabulary():
    """
    Maps words to indices and vice versa with standard index notation.
    Returns `None` when a word cannot be found, and throws KeyError
    when an index is out of range.

    New words can be added with `addWord()`.

    Uses `nltk.word_tokenize`.

    Example:
    ```
    >>> vocabulary = Vocabulary(["i'm", "a", "teapot"])
    >>> print(vocabulary["a"])
    5
    >>> print(vocabulary["supercalifragilisticexpialidocious"])
    None
    >>> print(vocabulary[6])
    "teapot"
    >>> print(vocabulary.count) # Includes the special tokens (e.g. <sos>)
    6
    ```
    """

    def __init__(self,
        tokens: Iterable[str],
        specialTokens: Union[Iterable[str], None] = [Token.SOS, Token.EOS, Token.UNK]
    ):

        self.wordToIndex = {k:v for v,k in enumerate(specialTokens)}
        self.indexToWord = [k for k,v in self.wordToIndex.items()]

        # Add all tokens to the vocabulary and check for duplicates
        for token in tokens:
            self.addWord(token)


    @classmethod
    def fromCorpus(classObject, corpus: str):
        tokens = tokenize(corpus)
        return classObject(tokens)


    def addWord(self, word: str):
        """Adds a word to the vocabulary if not already in vocabulary. Ignores ''.

        Args:
            word (str): New vocabulary word.
        """
        if word == '':
            return

        if word in self.wordToIndex:
            return

        index = len(self.indexToWord)

        self.wordToIndex[word] = index
        self.indexToWord.append(word)


    def matricize(self, sentence):
        """Converts a sentence into a one-hot matrix of size (word count, vocabulary size)"""

        # NOTE: `tokenize()` takes extra parameters, but the defaults are used by this function.
        # TODO ? : Find a solution to above problem?
        count = len(self.indexToWord)

        return oneHotify([self[token] for token in tokenize(sentence)], count)


    def __len__(self):
        return len(self.indexToWord)


    def __getitem__(self, key: Union[str, int, slice]):

        if type(key) is str:
            if key not in self.wordToIndex:
                return KeyError(f"'{key}' not in vocabulary!")

            return self.wordToIndex[key]

        elif type(key) is int:
            return self.indexToWord[key]

        else:
            raise KeyError(f"Vocabulary must be indexed via int or str, not '{type(key)}'")


class Corpus:
    def __init__(self, path, tokenized = False):
        self.path = path
        self.tokenized = tokenized

    def __iter__(self):
        for line in open(self.path):
            if self.tokenized:
                # Assumes tokens separated by spaces
                yield line.split(' ')
            else:
                yield tokenize(line)


def corpus2sentences(corpus): # -> Iterable[Iterable[str]]
    pass

def documents2sentences(documents): # -> Iterable[Iterable[str]]
    pass

# Initialize the model from an iterable of sentences. Each sentence
# is a list of words (unicode strings) that will be used for training.

# The sentences iterable can be simply a list, but for larger corpora,
# consider an iterable that streams the sentences directly from
# disk/network.

def transferLearning():
    word2vec = Word2Vec(
        [['testing', 'is', 'fun']],     # Iterable[Iterable[str]] where the strings are tokens
        sg= 1,        # 0: Continuous BOW | 1: skip-gram
        size= 50,     # Dimension of the word embedding vectors
        iter= 5,      # Epochs over the corpus
        window= 5,    # Radius of skip-gram / cbow window from current word
        min_count= 1, # Total frequency cut-off
    )

    more_sentences = [
        ['Advanced', 'users', 'can', 'load', 'a', 'model',
        'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']
    ]

# word2vec.save("wikipedia-2.word2vec")

# word2vec = Word2Vec.load("wikipedia-2.word2vec")

# word2vec = KeyedVectors.load_word2vec_format("Embedding/glove.6B/glove.6B.100d.txt")

# print(word2vec.most_similar('pilot'))

# model = KeyedVectors.load_word2vec_format("embeddings/crawl-300d-2M.vec")


def learnAmazonEmbedding(pretrainedFile):
    pretrainedPath = "embeddings/" + pretrainedFile

    print("Loading pretrained file ... (this may take several minutes)")

    # Word2Vec.load(pretrainedPath)

    print(f"Loaded '{pretrainedFile}'!")

    word2vec = Word2Vec(
        None,         # Union[Iterable[Iterable[str]], None] List of sentences containing lists of string tokens
        sg= 1,        # 0: Continuous BOW | 1: skip-gram
        size= 50,     # Dimension of the word embedding vectors
        iter= 5,      # Epochs over the corpus
        window= 5,    # Radius of skip-gram / cbow window from current word
        min_count= 1, # Total frequency cut-off
    )

    word2vec.build_vocab(more_sentences, update=True)
    word2vec.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)


def printUsage():
    print("Error!")
    print(f"  Usage: python {sys.argv[0]} <amazon/reddit> <embedding>")


def main():
    # corpus = Corpus("corpora/wikitext-2/wiki.train.tokens")
    # corpus = Corpus("corpora/wikipedia-encoding-article.txt")

    # word2vec = Word2Vec(
    #     corpus,       # Union[Iterable[Iterable[str]], None] List of sentences containing lists of string tokens
    #     sg= 1,        # 0: Continuous BOW | 1: skip-gram
    #     size= 50,     # Dimension of the word embedding vectors
    #     iter= 5,      # Epochs over the corpus
    #     window= 5,    # Radius of skip-gram / cbow window from current word
    #     min_count= 2, # Total frequency cut-off
    # )

    # matrix = word2vec.wv.vectors
    # index2word = word2vec.wv.index2word

    # print(index2word)

    # word2index = {word: index for index, word in enumerate(word2vec.wv.index2word)}
    # print(word2index
    #

    print(loadPretrained("embeddings/glove.6B.50d.vec"))

if __name__ == "__main__":

    if len(sys.argv) < 3:
        # printUsage()
        # exit()

        main()
        exit()

    dataSet = sys.argv[1]

    if dataSet not in ["amazon", "reddit"]:
        printUsage()
        exit()

    pretrainedFile = sys.argv[2]

    if dataSet == "amazon":
        learnAmazonEmbedding(pretrainedFile)

    if dataSet == "reddit":
        print("Reddit not yet implemented :(")