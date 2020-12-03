from shakespeare import shakespeareCorpus

from vocab import Vocabulary
from nltk import word_tokenize as tokenize



if __name__ == "__main__":

    corpus = shakespeareCorpus()

    vocabulary = Vocabulary.fromCorpus(corpus)

    # Short little demo
    print(vocabulary["the"])
    print(vocabulary[139])
    print(len(vocabulary))

    sentence = "Oh brother where art thou?"

    sentenceMatrix = vocabulary.matricize(sentence)
    print(sentenceMatrix.shape)