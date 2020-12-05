import os
from collections import Counter
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import time
import sys
import numpy as np
import spacy


def sequence(ratings, reviews, tw):
    seq = []
    for i, review in enumerate(reviews):
        for j in range(len(review)-tw):
            t_seq = ratings[i]+review[j:j+tw]
            t_label = review[j+tw]
            seq.append()

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
        
        #Initialize vocabulary and build vocab
        

        #self.s3 = boto3.resource('s3', region_name='us-west-2')
        
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        rating = self.rating[idx]
        review = self.review[idx]       
        
        #numericalize the review text
        
        
        return torch.tensor(review_vec), torch.tensor()

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

