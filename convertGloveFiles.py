import os
from gensim.scripts.glove2word2vec import glove2word2vec

# Convert the glove files to the correct format for gensim word2vec if needed
if not os.path.exists("embeddings/glove.42B.300d.vec"):
    glove2word2vec("embeddings/glove.42B.300d.txt", "embeddings/glove.42B.300d.vec")

if not os.path.exists("embeddings/glove.840B.300d.vec"):
    glove2word2vec("embeddings/glove.840B.300d.txt", "embeddings/glove.840B.300d.vec")

# Remove the original files once the .vec are created
if os.path.exists("embeddings/glove.42B.300d.vec") and os.path.exists("embeddings/glove.42B.300d.txt"):
    os.remove("embeddings/glove.42B.300d.txt")

if os.path.exists("embeddings/glove.840B.300d.vec") and os.path.exists("embeddings/glove.840B.300d.txt"):
    os.remove("embeddings/glove.840B.300d.txt")
