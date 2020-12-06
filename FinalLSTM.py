import os, time, sys, io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from embedding import loadPretrained, tokenize, Token
import glob
# EXAMPLE of using the embedding functions:
#
# word2index, index2word, matrix = loadPretrained("embeddings/glove.6B.300d.vec")
# padIndex = word2index(Token.PAD)
# words = tokenize("The dog sat on the ball.")
# embedded = torch.tensor([matrix[word2index[w]] for w in words])
#


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


def streamReviewsFromCSV(path):
    with open(path) as csv:
        for line in csv.readlines():
            df = pd.read_csv(io.StringIO(line), header=None)
            review = df[1][0]
            rating = df[0][0]

            if isNA(review) or isNA(rating) or type(rating) is not np.float64:
                continue

            if type(review) is not str:
                review = str(review)

            yield int(rating), tokenize(review)


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
            for rating, review in streamReviewsFromCSV(path):
                # Map words to indices in the embedding matrix
                indices = [self.vocabulary[Token.SOS]]
                indices += [
                    (
                        self.vocabulary[word]
                        if word in self.vocabulary
                        else self.vocabulary[Token.UNK]
                    )
                    for word in review
                ]
                indices.append(self.vocabulary[Token.EOS])

                for sequence, label in self.createWindows(indices):
                    # Tack the rating on to the front of the sequence
                    oneHotRating = int2vector(rating)

                    # NOTE: In order to use batches `len(sequence)` must always equal `windowSize`
                    yield torch.FloatTensor(oneHotRating), torch.tensor(sequence), torch.tensor(label)


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
        print("preEmbedding: ", preEmbedding.device)
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
        runningLoss = 0.0

        for (ratings, inputs, labels) in trainLoader:

            ratings = ratings.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Start with a blank state
            state = (
                torch.zeros(net.numLayers, trainLoader.batch_size, net.hiddenSize).double().to(device),
                torch.zeros(net.numLayers, trainLoader.batch_size, net.hiddenSize).double().to(device)
            )

            outputs = net(ratings, inputs, state)

            # Converts the labels into word vectors in embedding space
            labelVectors = net.embeddings(labels)

            loss = criterion(outputs, labelVectors)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {runningLoss}")

        train_loss_hist.append(runningLoss)
        epoch_hist.append(epoch)

    return train_loss_hist, epoch_hist


def main():
    #Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    embedpath = os.path.join(os.path.dirname(__file__), "embeddings/test.vec")

    print(f"Loading pretrained word2vec '{embedpath}'")

    word2index, index2word, matrix = loadPretrained(embedpath)

    # Dataloader parameters
    BATCH_SIZE = 4
    # Can't parallelize loading because we have a stream.
    # We could run a `create windows` script on our `.csv`s to get around this.
    NUM_WORKER = 0

    path = "data/amazon/csv/train/"
    trainingset = AmazonStreamingDataset(directory=path, windowSize=5, vocabulary=word2index)

    # Token to represent the padding
    padIndex = word2index[Token.PAD]

    dataLoader = DataLoader(
        dataset=trainingset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        collate_fn=CapsCollate(padIndex=padIndex)
    )

    net = RNN(
        inputSize= 300,
        hiddenSize= 100,
        numLayers= 2,
        preEmbedding= torch.tensor(matrix)
    ).double().to(device)

    train_loss, epoch = train(net, dataLoader, device, epochs=10)

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()

    # ax1.scatter(epoch, train_loss, label='training loss')
    # ax1.scatter(epoch, v_loss, label='validation loss')
    # plt.title('loss VS Epoch', fontsize=14)
    # plt.xlabel('epoch', fontsize=14)
    # plt.ylabel('loss', fontsize=14)
    # plt.grid(True)
    # plt.legend(loc='upper right')
    # plt.show()


if __name__ == "__main__":
    main()
