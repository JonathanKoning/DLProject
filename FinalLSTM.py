import os, time, sys, io, glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from gensim.models import KeyedVectors, Word2Vec
from nltk import word_tokenize


def tokenize(inputString):
    return word_tokenize(inputString.lower())


def loadPretrained(filePath):
    """Opens a `.vec` file and returns word2index, index2word,
    and the matrix of the word embedding vectors"""

    wordVectors = KeyedVectors.load_word2vec_format(filePath)

    vectorSize = wordVectors.vector_size

    # This assigns [0, 0, ..., 0] to SOS, EOS, and PAD, and
    # [random values] to UNK if they are not in the pretrained data.
    wordVectors.add(
        entities= [Token.SOS, Token.EOS, Token.PAD, Token.UNK],
        weights= [np.zeros(vectorSize), np.zeros(vectorSize), np.zeros(vectorSize), np.random.rand(vectorSize)],
        replace= False # Keep any existing vectors for these keys -- i.e. don't overwrite.
    )

    word2index = {word: index for index, word in enumerate(wordVectors.index2word)}

    return word2index, wordVectors


# Enumeration
class Token():
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"
    PAD = "<pad>"


def sequence(ratings, reviews, tw):
    seq = []
    for i, review in enumerate(reviews):
        for j in range(len(review)-tw):
            t_seq = ratings[i]+review[j:j+tw]
            t_label = review[j+tw]
            seq.append()


def int2vector(value):
    ar = np.zeros(300)
    ar[int(value)] = 1
    return ar


def isNA(value):
    return not pd.notna(value)


# Here's a quick sketch (julian) threw together for how this might look. PLEASE modify
class AmazonStreamingDataset(IterableDataset):
    def __init__(self, directory, windowSize, vocabulary):
        super().__init__()

        self.dataPaths = self.loadPaths(directory)
        self.windowSize = windowSize
        self.vocabulary = vocabulary

    # Iterates over all of the training data paths
    def loadPaths(self, directory):
        # Add absolute path to relative path
        datasetDirectory = os.path.join(os.path.dirname(__file__), directory)
        dataPaths = glob.glob(datasetDirectory + "*.csv")

        # Make sure the user didn't make any mistakes
        assert dataPaths != [], "Failed to locate training data!"

        return dataPaths

    # Generates windows of the review text on the fly via yield
    def createWindows(self, tokens):
        # NOTE: This does NOT create sequences shorter than `windowSize`
        for startIndex in range(len(tokens) - self.windowSize):
            endIndex = startIndex + self.windowSize

            yield tokens[startIndex:endIndex], tokens[endIndex]


    def __iter__(self):
        """Generates training/test examples in a stream.

        Yields:
            (sequence, label) [(list[str], str)]: `windowSize` words and the following target word.
        """
        for path in self.dataPaths:

            print(f"Reading '{path}'...", end='')
            df = pd.read_csv(path, header=None)
            print(f" done.")

            for index, row in df.iterrows():
                rating, tokens, label = row[0], row[1], row[2]

                # Map words to indices in the embedding matrix
                indices = [self.vocabulary[Token.SOS]]
                indices += [
                    (
                        self.vocabulary[word]
                        if word in self.vocabulary
                        else self.vocabulary[Token.UNK]
                    )
                    for word in tokenize(review)
                ]
                indices.append(self.vocabulary[Token.EOS])

                for sequence, label in self.createWindows(indices):
                    # Tack the rating on to the front of the sequence
                    oneHotRating = int2vector(rating)

                    # NOTE: In order to use batches `len(sequence)` must always equal `windowSize`
                    yield torch.FloatTensor(oneHotRating), torch.tensor(sequence), torch.tensor(label)


class AmazonDataset(Dataset):
    def __init__(self, path, vocabulary):
        super().__init__()

        self.vocabulary = vocabulary

        self.dataFrame = pd.read_csv(
            path,
            header= None,
            index_col= None,
            names= ['rating', 'tokens', 'target']
        )


    def __len__(self):
        return len(self.dataFrame)


    def __getitem__(self, index):
        row = self.dataFrame.iloc[index]

        rating, tokenString, label = row[0], row[1], row[2]

        # Already tokenized, just split on spaces
        tokens = tokenString.split(' ')

        # Convert sequence tokens to indices in vocabulary
        sequence = [
            (
                self.vocabulary[token]
                if token in self.vocabulary

                else self.vocabulary[Token.UNK]
            )
            for token in tokens
        ]

        # Convert label to index in vocabulary
        label = self.vocabulary[label] if label in self.vocabulary else self.vocabulary[Token.UNK]

        # Convert rating to one-hot vector
        oneHotRating = int2vector(rating)

        return torch.FloatTensor(oneHotRating), torch.tensor(sequence), torch.tensor(label)


# Applies padding to the reviews with the dataloader so that the reviews are all the same length.
class CapsCollate:

    def __init__(self, padIndex):
        self.padIndex = padIndex

    def __call__(self, batch):
        """ Converts a batch of inputs and outputs pairs, to a
        pair of batched inputs and batched outputs with padding
        applied to the inputs.
        """
        ratings = []
        sequences = []
        targets = []

        for (rating, sequence, target) in batch:
            ratings.append(rating.unsqueeze(0)) # Need to add a dimension so cat works later
            sequences.append(sequence)
            targets.append(target)

        paddedSequences = pad_sequence(sequences, batch_first=True, padding_value=self.padIndex)

        return torch.cat(ratings).double(), paddedSequences, torch.tensor(targets)


class RNN(nn.Module):

    def __init__(self, inputSize, hiddenSize, numLayers, preEmbedding):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(preEmbedding)
        # Turn off gradients--this means the embeddings cannot learn
        self.embeddings.requires_grad_ = False

        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        self.rnn = nn.LSTM(input_size=inputSize,
                           hidden_size=hiddenSize,
                           num_layers=numLayers,
                           batch_first=True)

        # Maps hidden state to tag
        self.fc = nn.Linear(hiddenSize, inputSize)


    def forward(self, rating, review, state):
        embeds = self.embeddings(review)
        inputs = torch.cat([rating.unsqueeze(1), embeds], dim=1)

        x, _ = self.rnn(inputs, state)

        # Take only the final output from the LSTM
        lastOutput = x[:,-1,:]

        x = self.fc(lastOutput)

        return x


def judgeAccuracy(outputs, labels, wordVectors, n):
    accurate = 0

    outputs = outputs.detach().numpy()
    labels  = labels.detach().numpy()

    # print(np.isfinite(outputs).all(), np.isfinite(labels).all())

    for (prediction, label) in zip(outputs, labels):
        correctToken = wordVectors.index2word[label]
        predictedTokens = [word for word, _ in wordVectors.similar_by_vector(prediction, topn=n)]

        if correctToken in predictedTokens:
            accurate += 1

    return accurate


def onehot2index(vector):
    for i, value in enumerate(vector):
        if value > 0:
            return i


def test(net, wordVectors, testLoader, device, n=3):
    runningLoss = 0.0
    accuracies = 0

    criterion = nn.MSELoss()

    for (ratings, inputs, labels) in testLoader:
        ratings = ratings.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)

        sizeOfBatch = labels.size(0)

        # Start with a blank state
        state = (
            torch.zeros(net.numLayers, sizeOfBatch, net.hiddenSize).double().to(device),
            torch.zeros(net.numLayers, sizeOfBatch, net.hiddenSize).double().to(device)
        )

        outputs = net(ratings, inputs, state)

        # Converts the labels into word vectors in embedding space
        labelVectors = net.embeddings(labels)

        loss = criterion(outputs, labelVectors)
        lossValue = loss.item()
        runningLoss += lossValue

        # ratings = ratings.detach().to("cpu")
        # inputs = inputs.detach().to("cpu")
        labels = labels.detach().to("cpu")
        outputs = outputs.detach().to("cpu")

        accuracies += judgeAccuracy(outputs, labels, wordVectors, n)

        # Print a little demo to the screen
        # for (rating, inputSequence, label, prediction) in zip(ratings, inputs, labels, outputs):
        #     stars = onehot2index(rating)
        #     print("Prompt:", f"<{stars} stars>", [wordVectors.index2word[i] for i in inputSequence])
        #     print("Predictions:", [word for word, _ in wordVectors.similar_by_vector(prediction.detach().numpy())])
        #     print("Label:", wordVectors.index2word[label])

    return runningLoss / len(testLoader), accuracies / len(testLoader)


def train(net, trainLoader, device, epochs=20):

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Couldn't get perplexity to work. Cross-entropy loss doesn't apply since we're not using classes.
    # TODO: Find correct loss function to use for this task.
    criterion = nn.MSELoss()

    train_loss_hist = []
    train_acc_hist = []
    epoch_hist = []
    val_loss_hist = []
    val_acc_hist = []

    for epoch in range(epochs):

        # print(f"Epoch {epoch + 1}")
        epochLoss = 0.0

        for (ratings, inputs, labels) in trainLoader:

            ratings = ratings.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            sizeOfBatch = labels.size(0)

            optimizer.zero_grad()

            # Start with a blank state
            state = (
                torch.zeros(net.numLayers, sizeOfBatch, net.hiddenSize).double().to(device),
                torch.zeros(net.numLayers, sizeOfBatch, net.hiddenSize).double().to(device)
            )

            outputs = net(ratings, inputs, state)

            # Converts the labels into word vectors in embedding space
            labelVectors = net.embeddings(labels)

            loss = criterion(outputs, labelVectors)
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {epochLoss / len(trainLoader)}")

        torch.save(net.state_dict(), f"model-epoch-{str(epoch + 1)}.torch")

        train_loss_hist.append(epochLoss)
        epoch_hist.append(epoch)

    return train_loss_hist, epoch_hist


def main():

    embedpath = os.path.join(os.path.dirname(__file__), "trained/prime-pantry-300d.vec")
    print(f"Loading pretrained word2vec '{embedpath}'...", end='')

    word2index, wordVectors = loadPretrained(embedpath)
    print(" done.")



    #Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # Dataloader parameters
    BATCH_SIZE = 1024
    # Parallelize away!
    NUM_WORKER = 2

    trainingPath = "/content/drive/Shareddrives/DLFinalProject/data/prime-pantry-10k-train.csv"
    testingPath = "/content/drive/Shareddrives/DLFinalProject/data/prime-pantry-10k-test.csv"
    trainingSet = AmazonDataset(trainingPath, vocabulary=word2index)
    testingSet = AmazonDataset(testingPath, vocabulary=word2index)

    # Token to represent the padding
    padIndex = word2index[Token.PAD]

    trainLoader = DataLoader(
        dataset=trainingSet,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        collate_fn=CapsCollate(padIndex=padIndex)
    )

    testLoader = DataLoader(
        dataset=testingSet,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        collate_fn=CapsCollate(padIndex=padIndex)
    )

    net = RNN(
        inputSize= 300,
        hiddenSize= 1024,
        numLayers= 2,
        preEmbedding= torch.tensor(wordVectors.vectors)
    ).double().to(device)

    train_loss, epoch = train(net, trainLoader, device, epochs=100)

    print("Train loss/accuracy")
    print(test(net, wordVectors, trainLoader, device))
    print("Test loss/accuracy")
    print(test(net, wordVectors, testLoader, device))



if __name__ == "__main__":
    main()