import io, glob, os, string, sys

import numpy as np
from pandas import read_csv, notna

from collections import Counter
from random import shuffle
from gensim.models import KeyedVectors, Word2Vec

from tokenizer import nltk_tokenize, tokenize
from LSTM import Token


def currentDirectory():
    return os.path.dirname(__file__)


def loadSentencesFromCSV(path, withoutTokens=False):
    sentences = []
    df = read_csv(path, header=None)

    for index, row in df.iterrows():
        review = row[1]

        if not notna(review):
            continue

        if type(review) is not str:
            review = str(review)

        if withoutTokens:
            sentences.append(review)
        else:
            sentences.append(tokenize(review))

    return sentences


def loadSentencesFromMultiple(paths):
    sentences = []
    for path in paths:
        sentences += loadSentencesFromCSV(path)

    shuffle(sentences)

    return sentences


def learnEmbeddings(paths, outputName, minCount):

    print(f"Creating word2vec model...")

    sentences = loadSentencesFromMultiple(paths)

    print(f"{len(sentences)} sentences loaded.")

    word2vec = Word2Vec(
        sentences,
        sg= 1,
        size= 300,         # Dimension of the word embedding vectors
        window= 5,    # Radius of skip-gram / cbow window from current word
        min_count= minCount,
        iter= 5
    )

    word2vec.wv.save_word2vec_format("embeddings/" + outputName + "-300d.vec")


def food():
    paths = [
        "data/csv/grocery-and-gourmet-food-test.csv",
        "data/csv/grocery-and-gourmet-food-train.csv",
        "data/csv/prime-pantry-train.csv",
    ]

    outputName = "food-lower"

    learnEmbeddings(paths, outputName, minCount=3)



def allMin5():
    paths = [
        "data/csv/all-beauty-test.csv",
        "data/csv/all-beauty-train.csv",
        "data/csv/grocery-and-gourmet-food-test.csv",
        "data/csv/grocery-and-gourmet-food-train.csv",
        "data/csv/home-and-kitchen-test.csv",
        "data/csv/home-and-kitchen-train.csv",
        "data/csv/office-products-train.csv",
        "data/csv/office-products-test.csv",
        "data/csv/pet-supplies-train.csv",
        "data/csv/pet-supplies-test.csv",
        "data/csv/prime-pantry-train.csv",
    ]

    outputName = "all-lower-min5"

    learnEmbeddings(paths, outputName, minCount=5)


def allMin3():
    paths = [
        "data/csv/all-beauty-test.csv",
        "data/csv/all-beauty-train.csv",
        "data/csv/grocery-and-gourmet-food-test.csv",
        "data/csv/grocery-and-gourmet-food-train.csv",
        "data/csv/home-and-kitchen-test.csv",
        "data/csv/home-and-kitchen-train.csv",
        "data/csv/office-products-train.csv",
        "data/csv/office-products-test.csv",
        "data/csv/pet-supplies-train.csv",
        "data/csv/pet-supplies-test.csv",
        "data/csv/prime-pantry-train.csv",
    ]

    outputName = "all-lower-min3"

    learnEmbeddings(paths, outputName, minCount=3)


if __name__ == "__main__":

    food()

    allMin3()

    allMin5()

    vectors = KeyedVectors.load_word2vec_format()