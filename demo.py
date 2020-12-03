from shakespeare import shakespeareCorpus

from vocab import tokenize, Vocabulary


if __name__ == "__main__":

    corpus = shakespeareCorpus()

    # Ignore special characters that aren't common parts of speech
    special = ("*+/<=>@\^_`{|}~")

    tokens = tokenize(corpus, ignoreCharacters = special)
    vocabulary = Vocabulary(tokens)

    # Short little demo
    print(vocabulary["the"])
    print(vocabulary[139])
    print(len(vocabulary))

    sentence = "Oh brother where art thou?"

    sentenceMatrix = vocabulary.matricize(sentence)
    print(sentenceMatrix.shape)