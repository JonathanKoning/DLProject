import os

from nltk import word_tokenize as tokenize
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec


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


# Initialize the model from an iterable of sentences. Each sentence
# is a list of words (unicode strings) that will be used for training.

# The sentences iterable can be simply a list, but for larger corpora,
# consider an iterable that streams the sentences directly from
# disk/network.

def transferLearning():
    word2vec = gensim.models.Word2Vec(
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

    word2vec.build_vocab(more_sentences, update=True)
    word2vec.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)

# word2vec.save("wikipedia-2.word2vec")

# word2vec = Word2Vec.load("wikipedia-2.word2vec")

# word2vec = KeyedVectors.load_word2vec_format("Embedding/glove.6B/glove.6B.100d.txt")

# print(word2vec.most_similar('pilot'))

# model = KeyedVectors.load_word2vec_format("embeddings/crawl-300d-2M.vec")


if __name__ == "__main__":
