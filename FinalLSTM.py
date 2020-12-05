import os, time, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from embedding import loadPretrained, tokenize, Token

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

def inttovector(value):
    ar = np.zeros(300)
    ar[int(value)] = 1
    return ar


class AmazonDataset(Dataset):
    """
    AmazonDataset
    """
    def __init__(self,path):
        #read in dataset in json format and convert to csv
        self.df = pd.read_json(path, lines=True)
        self.df = df.to_csv()

        #Get rating and review data
        self.rating = self.df["overall"]
        self.review = self.df["reviewText"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        rating = self.rating[idx]
        review = self.review[idx]

        #numericalize the review text

        return torch.tensor(review_vec), torch.tensor()


# Here's a quick sketch (julian) threw together for how this might look. PLEASE modify
class AmazonStreamingDataset(IterableDataset):
    def __init__(self, directory, windowSize, train=False):
        super().__init__()

        self.directory = directory
        self.windowSize = windowSize
        self.isTraining = train


    # Iterates over all of the training data paths
    def paths(self):
        #add absolute path to relative path
        datasetPaths = os.path.join(os.path.dirname(__file__), self.directory) 

        # TODO: Sort paths so 1 is before 9 is before 10 is before ...
        # NOTE: 10 is often sorted before 9 in filesystems, so double check!

        # TODO: Testing / training split using `self.isTraining`

        for path in glob.glob(datasetPaths)):
            yield path


    # Generates windows of the review text on the fly via yield
    def createWindows(rating, reviewText):
        tokens = tokenize(reviewText)
    
        vrating = inttovector(rating)
        for j in range(len(tokens)-self.windowSize):
            sequence = vrating+tokens[j:j+self.windowSize]
            label = tokens[j+self.windowSize]

            yield sequence, label


    # Generates training/test examples in a stream.
    def __iter__(self):
        for path in self.paths():
            # Only load one file at a time to conserve memory
            file = np.load_csv(path)

            for row in file:
                # TODO: Fix up this call to work with whatever representation
                for sequence, label in createWindows(row["rating"], row["review"]):
                    yield sequence, label


class CapsCollate:
  #Applys padding to the reviews with the dataloader so that the reviews are all the same length
  def __init__(self,pad_idx,batch_first=False):
    self.pad_idx = pad_idx
    self.batch_first = batch_first

  def __call__(self,batch):
    #TODO
    #Do something with the ratings here

    targets = [item[1] for item in batch]
    targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)

    #return ratings, targets
    return targets


#TODO
class RNN(nn.Module):
    def __init__(self, input_size, hidden_units, layers_num, pre_embedding):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(pre_embedding)
        self.embeddings.requires_grad_ = False

        self.rnn = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=layers_num,
                            batch_first=True)

        #maps hidden state to tag
        self.fc = nn.Linear(hidden_units, input_size)

    def forward(self, review, state=None):
        embeds = self.embeddings(review)
        x, _ self.rnn(embeds, state)
        x = self.out(x)

        return x

def train(net, train_loader, epochs=20):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loss_hist = []
    train_acc_hist = []
    epoch_hist = []
    val_loss_hist = []
    val_acc_hist = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs.to(device)
            labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)

            #apply perplexity loss
            loss = torch.exp(criterion(outputs))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss_hist.append(running_loss)
        epoch_hist.append(epoch)

    return train_loss_hist, epoch_hist


def main():
    #Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    #Dataloader
    BATCH_SIZE = 4
    NUM_WORKER = 1

    path = ""
    dataset = AmazonDataset(path=path)
    #token to represent the padding
    pad_idx = dataset.vocab.stoi["<PAD>"]

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
        colate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
    )

    net = RNN().to(device)
    train_loss, epoch = train(net, data_loader, 125)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()

    ax1.scatter(epoch, train_loss, label='training loss')
    #ax1.scatter(epoch, v_loss, label='validation loss')
    plt.title('loss VS Epoch', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()