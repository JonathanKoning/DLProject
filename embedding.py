import os, sys, io, glob
from typing import Union, Iterable

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec

from FinalLSTM import tokenize, Token, loadPretrained

from random import shuffle

def currentDirectory():
    return os.path.dirname(__file__)


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


def loadSentencesFromCSV(path):
    sentences = []
    df = pd.read_csv(path, header=None)

    for index, row in df.iterrows():
        review = row[1]

        if not pd.notna(review):
            continue

        if type(review) is not str:
            review = str(review)

        sentences.append(tokenize(review))

    return sentences


def loadSentencesFromMultiple(paths):
    sentences = []
    for path in paths:
        sentences.append(loadSentencesFromCSV(path))

    shuffle(sentences)

    return sentences


def learnAmazonEmbedding(architecture="sg", window=5, epochs=5, minFreq=2):

    availableArchitectures = ['sg', 'cbow']
    output = ", ".join(availableArchitectures)
    assert architecture in availableArchitectures, f"'architecture' must be either {output}!"

    model = 1 if architecture == 'sg' else 0 # 0: Continuous BOW | 1: skip-gram

    trainDirectory = os.path.join(currentDirectory(), "data/amazon/csv/")
    trainPaths = glob.glob(trainDirectory + "*_5.csv")

    trainPaths = sorted(trainPaths, key=os.path.getsize)

    assert trainPaths != [], "Unable to find training files!"

    for path in trainPaths:
        fileName = os.path.basename(path)
        outputName = (fileName.lower()[:-6]).replace('_', '-')

        print(f"Creating word2vec model for '{path}' using '{architecture}'...")

        word2vec = Word2Vec(
            loadSentencesFromCSV(path),
            sg= model,
            size= 300,         # Dimension of the word embedding vectors
            window= window,    # Radius of skip-gram / cbow window from current word
            min_count= minFreq,
            iter= epochs
        )

        word2vec.save("model/" + outputName + "-300d.model")
        word2vec.wv.save_word2vec_format("trained/" + outputName + "-300d.vec")


def learnFoodEmbeddings():
    paths = ["data/amazon/csv/Grocery_and_Gourmet_Food_5.csv", "data/amazon/csv/Prime_Pantry_5.csv"]

    outputName = "food"

    print(f"Creating word2vec model...")

    word2vec = Word2Vec(
        loadSentencesFromMultiple(paths),
        sg= 1,
        size= 300,         # Dimension of the word embedding vectors
        window= 5,    # Radius of skip-gram / cbow window from current word
        min_count= 2,
        iter= 5
    )

    word2vec.save("model/" + outputName + "-300d.model")
    word2vec.wv.save_word2vec_format("trained/" + outputName + "-300d.vec")


def learnFoodAndHouseEmbeddings():
    paths = ["data/amazon/csv/Grocery_and_Gourmet_Food_5.csv", "data/amazon/csv/Prime_Pantry_5.csv", "data/amazon/csv/Home_and_Kitchen_5.csv"]

    outputName = "food-and-home"

    print(f"Creating word2vec model...")

    word2vec = Word2Vec(
        loadSentencesFromMultiple(paths),
        sg= 1,
        size= 300,         # Dimension of the word embedding vectors
        window= 5,    # Radius of skip-gram / cbow window from current word
        min_count= 5,
        iter= 3
    )

    word2vec.save("model/" + outputName + "-300d.model")
    word2vec.wv.save_word2vec_format("trained/" + outputName + "-300d.vec")



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

    # learnAmazonEmbedding()

    learnFoodAndHouseEmbeddings()

    # word2vec = Word2Vec.load("model/grocery-and-gourmet-food-300d.model")

    # print(word2vec.wv.most_similar('bread'))