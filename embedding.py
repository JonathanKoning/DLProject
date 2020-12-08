import os, sys, io, glob

import numpy as np
import pandas as pd

from random import shuffle
from gensim.models import KeyedVectors, Word2Vec
from nltk import word_tokenize



def currentDirectory():
    return os.path.dirname(__file__)


def tokenize(inputString):
    return word_tokenize(inputString.lower())


def bespokeTokenize(inputString):

    # 1. Lower case
    # 2. Split on spaces and phrase-level punctuation
    # 3. What about token-level punctuation? (apostrophe)
    # 4. Fix mis-spellings
    # 5. Check if the token matches <num> / <date> / <url>

    # Don't split on dashes
    # Do split on slashes
    # Do split on period
    # Don't split on apostrophe's

    return word_tokenize(inputString.lower())


# Enumeration
class Token():
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"
    PAD = "<pad>"



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
        sentences += loadSentencesFromCSV(path)

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

    sentences = loadSentencesFromMultiple(paths)

    print(len(sentences))

    word2vec = Word2Vec(
        sentences,
        sg= 1,
        size= 300,         # Dimension of the word embedding vectors
        window= 5,    # Radius of skip-gram / cbow window from current word
        min_count= 2,
        iter= 5
    )

    word2vec.save("model/" + outputName + "-300d.model")
    word2vec.wv.save_word2vec_format("trained/" + outputName + "-300d.vec")


# home and kitchen is TOO big. may be able to downsample.
def learnFoodAndHouseEmbeddings():
    paths = ["data/amazon/csv/Grocery_and_Gourmet_Food_5.csv", "data/amazon/csv/Prime_Pantry_5.csv", "data/amazon/csv/Home_and_Kitchen_5.csv"]

    outputName = "food-and-home"

    print("Loading sentences...", end="")

    sentences = loadSentencesFromMultiple(paths)

    print(f"done. ({sentences})")

    print(f"Creating word2vec model...")

    word2vec = Word2Vec(
        sentences,
        sg= 1,
        size= 300,         # Dimension of the word embedding vectors
        window= 5,    # Radius of skip-gram / cbow window from current word
        min_count= 5,
        iter= 3
    )

    word2vec.save("model/" + outputName + "-300d.model")
    word2vec.wv.save_word2vec_format("trained/" + outputName + "-300d.vec")


if __name__ == "__main__":

    # learnAmazonEmbedding()

    learnFoodEmbeddings()

    # word2vec = Word2Vec.load("model/grocery-and-gourmet-food-300d.model")

    # print(word2vec.wv.most_similar('bread'))