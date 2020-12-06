import os, sys, io, glob
from typing import Union, Iterable

import numpy as np
import pandas as pd
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


def currentDirectory():
    return os.path.dirname(__file__)


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


# word2vec.save("wikipedia-2.word2vec")

# word2vec = Word2Vec.load("wikipedia-2.word2vec")

# word2vec = KeyedVectors.load_word2vec_format("Embedding/glove.6B/glove.6B.100d.txt")

# print(word2vec.most_similar('pilot'))

# model = KeyedVectors.load_word2vec_format("embeddings/crawl-300d-2M.vec")


class AmazonReviewStream(object):
    def __init__(self, paths, showEpochs=False):
        self.paths = paths
        self.showEpochs = showEpochs
        self.epoch = 0

    def __iter__(self):
        if self.showEpochs:
            self.epoch += 1
            print(f"  [Epoch {self.epoch}]")

        for path in self.paths:
            print(f"  {path}")

            with open(path) as csv:
                number = 0
                for line in csv.readlines():
                    number += 1

                    if number % 1000 == 0:
                        print(f"    {number} lines...", end="\r")

                    df = pd.read_csv(io.StringIO(line), header=None)
                    review = df[1][0]

                    if not pd.notna(review):
                        continue

                    if type(review) is not str:
                        review = str(review)

                    yield tokenize(review)


def learnAmazonEmbedding(architecture="sg", window=5, epochs=3, minFreq=5):

    availableArchitectures = ['sg', 'cbow']
    output = ", ".join(availableArchitectures)
    assert architecture in availableArchitectures, f"'architecture' must be either {output}!"

    model = 1 if architecture == 'sg' else 0 # 0: Continuous BOW | 1: skip-gram

    trainDirectory = os.path.join(currentDirectory(), "data/amazon/csv/")
    trainPaths = ["data/amazon/csv/1.csv"] # glob.glob(trainDirectory + "*.csv")

    assert trainPaths != [], "Unable to find training files!"

    print(f"\nCreating word2vec model using '{architecture}'...")

    # word2vec = Word2Vec(
    #     AmazonReviewStream(trainPaths),
    #     sg= model,
    #     size= 300,         # Dimension of the word embedding vectors
    #     iter= 3,
    #     window= window,    # Radius of skip-gram / cbow window from current word
    #     min_count= minFreq,
    # )

    word2vec = Word2Vec(
        sg= model,
        size= 300,         # Dimension of the word embedding vectors
        window= window,    # Radius of skip-gram / cbow window from current word
        min_count= minFreq,
    )

    print(f"\nBuilding vocabulary...")

    word2vec.build_vocab(AmazonReviewStream(trainPaths, showEpochs=False))

    print(f"\nTraining...")

    word2vec.train(AmazonReviewStream(trainPaths), total_examples=word2vec.corpus_count, epochs=epochs, report_delay=1.0)

    print(f"\Saving...")

    word2vec.save("amazon-300d.model")

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

    learnAmazonEmbedding("sg")