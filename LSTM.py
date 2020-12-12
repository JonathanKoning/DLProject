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


# Enumeration
class Token():
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"
    PAD = "<pad>"
    Number = "<num>"
    URL = "<num>"


def loadPretrained(filePath):
    """Opens a `.vec` file and returns word2index and
    wordVectors (KeyedVectors) holding the word embedding vectors"""

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


def int2vector(value):
    ar = np.zeros(300)
    ar[int(value)] = 1
    return ar


def isNA(value):
    return not pd.notna(value)


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

    def __init__(self, inputSize, hiddenSize, outputSize, numLayers, preEmbedding):
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
        self.fc = nn.Linear(hiddenSize, outputSize)


    def forward(self, rating, review, state):
        embeds = self.embeddings(review)
        inputs = torch.cat([rating.unsqueeze(1), embeds], dim=1)

        x, _ = self.rnn(inputs, state)

        # Take only the final output from the LSTM
        lastOutput = x[:,-1,:]

        x = self.fc(lastOutput) # `softmax` should not be added before `CrossEntropyLoss`

        return x


def judgeAccuracy(outputs, labels, wordVectors, n):
    accurate = 0

    outputs = outputs.detach().numpy()
    labels  = labels.detach().numpy()

    for (prediction, label) in zip(outputs, labels):
        correctToken = wordVectors.index2word[label]

        topIndices = np.argpartition(prediction, (-1) * n)[(-1) * n:]
        predictedTokens = [wordVectors.index2entity[i] for i in topIndices]

        # print(topIndices, predictedTokens)

        if correctToken in predictedTokens:
            accurate += 1

    return accurate


def onehot2index(vector):
    for i, value in enumerate(vector):
        if value > 0:
            return i


def test(net, wordVectors, testLoader, batchsize, device, n=3, output=False):
    runningLoss = 0.0
    accuracies = 0

    criterion = nn.CrossEntropyLoss()

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

        loss = criterion(outputs, labels)
        lossValue = loss.item()
        runningLoss += lossValue

        labels = labels.detach().to("cpu")
        outputs = outputs.detach().to("cpu")

        accuracies += judgeAccuracy(outputs, labels, wordVectors, n)

        if output:
            ratings = ratings.detach().to("cpu")
            inputs = inputs.detach().to("cpu")

            # Print a little demo to the screen
            for (rating, inputSequence, label, prediction) in zip(ratings, inputs, labels, outputs):
                stars = onehot2index(rating)
                print("Prompt:", f"<{stars} stars>", [wordVectors.index2word[i] for i in inputSequence])
                print("Predictions:", [word for word, _ in wordVectors.similar_by_vector(prediction.detach().numpy())])
                print("Label:", wordVectors.index2word[label])

    totalExamples = len(testLoader) * batchsize

    return runningLoss / totalExamples, accuracies / totalExamples


def train(net, wordVectors, trainLoader, testLoader, device, batchsize, epochs=20):

    optimizer = optim.Adam(net.parameters(), lr=0.003)

    # Couldn't get perplexity to work. Cross-entropy loss doesn't apply since we're not using classes.
    # TODO: Find correct loss function to use for this task.
    criterion = nn.CrossEntropyLoss()

    train_loss_hist = []
    train_acc_hist = []
    epoch_hist = []
    val_loss_hist = []
    val_acc_hist = []

    totalExamples = len(testLoader) * batchsize

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

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        epochLoss /= totalExamples

        print(f"Epoch {epoch + 1} Loss: {epochLoss}")

        if (epoch + 1) % 5 == 0:
            print("Test loss/accuracy", end='')
            print(test(net, wordVectors, testLoader, batchsize, device))

            torch.save(net.state_dict(), f"model-epoch-{str(epoch + 1)}.torch")

        train_loss_hist.append(epochLoss)
        epoch_hist.append(epoch)

    return train_loss_hist, epoch_hist


def main():

    ON_COLAB = False

    embedfile = "embeddings/all-lower-89k-300d.vec"
    trainfile = "data/prime-pantry-lower-200k-train.csv"
    testfile = "data/prime-pantry-lower-40k-test.csv"
    
    if ON_COLAB:
        embedpath = "/content/drive/Shareddrives/DLFinalProject/" + embedfile

    else:
        embedpath = os.path.join(os.path.dirname(__file__), embedfile)

    print(f"Loading pretrained word2vec '{os.path.basename(embedpath)}'...", end='')

    word2index, wordVectors = loadPretrained(embedpath)
    print(" done.")



    #Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # Dataloader parameters
    BATCH_SIZE = 256
    # Parallelize away!
    NUM_WORKER = 1

    HIDDEN_SIZE = 512

    if ON_COLAB:
        trainingPath = "/content/drive/Shareddrives/DLFinalProject/" + trainfile
        testingPath = "/content/drive/Shareddrives/DLFinalProject/" + testfile

    else:
        trainingPath = os.path.join(os.path.dirname(__file__), trainfile)
        testingPath = os.path.join(os.path.dirname(__file__), testfile)

    trainingSet = AmazonDataset(trainingPath, vocabulary=word2index)
    testingSet = AmazonDataset(testingPath, vocabulary=word2index)

    print("EmbedPath: ", embedfile)
    print("TrainPath: ",trainfile)
    print("TestPath: ", testfile)
    print("BatchSize: ", BATCH_SIZE)
    print("Workers: ", NUM_WORKER)
    print("Hidden_Size: ", HIDDEN_SIZE)
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
        hiddenSize= HIDDEN_SIZE,
        outputSize= len(wordVectors.index2word),
        numLayers= 1,
        preEmbedding= torch.tensor(wordVectors.vectors)
    ).double().to(device)

    # net.load_state_dict(torch.load("model.torch", map_location=torch.device('cpu')))

    train_loss, epoch = train(net, wordVectors, trainLoader, testLoader, device, BATCH_SIZE, epochs=40)

    print("Train loss/accuracy")
    print(test(net, wordVectors, trainLoader, BATCH_SIZE, device))
    print("Test loss/accuracy")
    print(test(net, wordVectors, testLoader, BATCH_SIZE, device))



if __name__ == "__main__":
    main()