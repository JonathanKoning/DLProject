"""
Reads json files using generators that return lists with only the information we care about.

Examples:

```
for comment in reddit():
    print(comment)

>>>
['gameofthrones', 't5_2rjz2', 4, 'i too vote in favor for more male nudity ']
...
```

```
for review in amazon("Gift_Cards_5.json"):
        print(review)
>>>
[5.0, 'Pretty good!']
...
```
"""


import datetime
import json
import string
import os
import glob

import pandas as pd
import numpy as np

from random import shuffle, random, choice
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from FinalLSTM import tokenize, Token, isNA

asciiCharacters = set(string.printable)


def currentDirectory():
    return os.path.dirname(__file__)


def clean(dirtyString):
        asciiString = "".join(list(filter(lambda c: c in asciiCharacters, dirtyString)))

        cleanString = asciiString.replace("\r", ' ').replace("\n", ' ')

        return cleanString


def cleanIfString(maybeString):
    if type(maybeString) is str:
        return clean(maybeString)
    else:
        return maybeString


def jsonToList(jsonData, fields):
    return [cleanIfString(jsonData[f]) if f in jsonData else None for f in fields]


def reddit():

    fields = ["subreddit", "subreddit_id", "score", "body"]

    with open("data/reddit/comments.json") as file:

        for line in file:
            commentData = json.loads(clean(line.strip()))

            body = commentData["body"]

            if body == "[deleted]":
                continue

            yield jsonToList(commentData, fields)


def createCSVFiles(n, relativeDirectory, headers):
    directory = os.path.join(currentDirectory(), relativeDirectory)

    # Create the directory if needed
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Just double check we're not doing something we don't mean to
    # because this takes a long time to rebuild the .csv's
    else:
        if input(f"WARNING! This will overwrite everything in '{directory}'\nIs that ok? (y): ") != 'y':
            print("Aborting.")
            exit(0)
        else:
            print("You got it chief! (☞ﾟヮﾟ)☞ Continuing...\n")

    # Create one csv for each bucket
    for csvNumber in range(n):
        fileNumber = csvNumber + 1

        fileName = str(fileNumber) + ".csv"
        path = os.path.join(directory, fileName)

        initialContents = ",".join(headers) + "\n"

        # Create file / overwrite with the pandas headers
        with open(path, 'w') as file:
            file.write(initialContents)


def splitAmazonData(n):
    """Allocates the Amazon .json data files into `n`
    .csv files where each Category has the same number of
    reviews in each new .csv file.

    Args:
        n (int): How many .csv files to generate.
    """

    # What are the JSON fields we want
    fields = ["overall", "reviewText"]
    # What do we want to call those columns in pandas
    columns = ["rating", "review"]

    path = "data/amazon/Prime_Pantry_5.json"

    reviews = []

    # Converts amazon review object to list using `fields`
    def jsonToRow(jsonData):
        return [(clean(str(jsonData[f])) if f in jsonData else "") for f in fields]

    with open(path) as file:
        for line in file:
            reviews.append(jsonToList(json.loads(line.strip()), fields))

    # Convert the data in a Dataframe for easy I/O
    reviewDataFrame = pd.DataFrame(reviews, columns=columns)

    # Go ahead and save this category version
    name = os.path.basename(path)[:-5] + ".csv"
    outPath = os.path.join(currentDirectory(), "data/amazon/csv/" + name)
    reviewDataFrame.to_csv(outPath)


def shuffleCSV(path):
    df = pd.read_csv(path, header=None)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    # Over-write the original file
    shuffled_df.to_csv(path, index=False, header=None, mode='w')


# Converts ratings to ints
def integerizeCSV(path):
    df = pd.read_csv(path, header=None, names=["rating", "review"])
    df = df.astype({"rating": int})
    df.to_csv(path, index=False, header=None, mode='w')


def shuffleAll(n):
    for outputIndex in range(n):
        name = str(outputIndex) + ".csv"
        outPath = os.path.join(currentDirectory(), "data/amazon/csv/" + name)

        print("Shuffling "+ outPath)

        shuffleCSV(outPath)


def createCategoryCSV(path):
    # What are the JSON fields we want
    fields = ["overall", "reviewText"]
    # What do we want to call those columns in pandas
    columns = ["rating", "review"]

    reviews = []

    # Converts amazon review object to list using `fields`
    def jsonToRow(jsonData):
        return [(clean(str(jsonData[f])) if f in jsonData else "") for f in fields]

    with open(path) as file:
        for line in file:
            reviews.append(jsonToList(json.loads(line.strip()), fields))

    # Convert the data in a Dataframe for easy I/O
    reviewDataFrame = pd.DataFrame(reviews, columns=columns)
    # Convert ratings to integers
    reviewDataFrame = reviewDataFrame.astype({"rating": int})

    # Go ahead and save this category version
    name = os.path.basename(path)[:-5] + ".csv"
    outPath = os.path.join(currentDirectory(), "data/amazon/csv/" + name)
    reviewDataFrame.to_csv(outPath, index=False, header=False)


# Visualize and output the occurences
def histogramRatings(path, datasetFile=False):
    names = ['rating', 'tokens', 'target'] if datasetFile else ['rating', 'review']
    df = pd.read_csv(path, header=None, index_col=None, names=names)

    print(Counter(df["rating"].to_numpy()))


def balanceRatings(path):
    df = pd.read_csv(path, header=None, index_col=None, names=['rating', 'review'])

    rus = RandomUnderSampler(sampling_strategy='not minority', random_state=1)
    df_balanced, balanced_labels = rus.fit_resample(df, df['rating'])
    shuffled_df = df_balanced.sample(frac=1).reset_index(drop=True)

    outPath = path[:-4] + "_Balanced.csv"

    df_balanced.to_csv(outPath, index=False, header=False)


def createCategoryAndBalancedCSVs():
    jsonDirectory = os.path.join(currentDirectory(), "data/amazon/*.json")
    for path in glob.glob(jsonDirectory):

        fileName = os.path.basename(path)
        directory = '/'.join(path.split('/')[:-1]) + '/'
        csvFilePath = directory + "csv/" + fileName.split('.')[0] + ".csv"
        balancedFilePath = csvFilePath.split('.')[0] + "_Balanced.csv"

        # createCategoryCSV(path)
        # shuffleCSV(csvFilePath)
        # balanceRatings(csvFilePath)
        # histogramRatings(balancedFilePath)

        print(csvFilePath)

        integerizeCSV(csvFilePath)

        print(balancedFilePath)

        integerizeCSV(balancedFilePath)


# Creates a list of windows of the review text
def createWindows(tokens, maxWindowSize, fullWindowsOnly=False, onlyChoseOne=False):
    windows = []

    beginningIndex = 0 if fullWindowsOnly else 1 - maxWindowSize

    startPositions = range(beginningIndex, len(tokens) - maxWindowSize)

    if onlyChoseOne:
        startPositions = [choice(startPositions)]

    for startIndex in startPositions:
        endIndex = startIndex + maxWindowSize

        startIndex = 0 if startIndex < 0 else startIndex
        endIndex = endIndex if endIndex < len(tokens) else len(tokens) - 1

        windows.append((tokens[startIndex:endIndex], tokens[endIndex]))

    return windows


def createDataset(path, outputPath, length, maxWindowSize=5, fullWindowsOnly=False, testFraction=0):
    """Creates a dataset that can be consumed by an indexed DataSet
    and balances classes.

    Args:
        path (str): What .csv to read from (prefer unbalanced)
        outputPath (str): Location for dataset
        length (int): Desired size of dataset
        maxWindowSize (int): How many words to include in window at most
        fullWindowsOnly (bool): Force all windows to be `maxWindowSize`
    """

    if length % 5 != 0:
        print(f"[WARNING] `length` ({length}) is not a multiple of 5! length={length//5 * 5} will be used instead.\n")

    ratingsFrame = pd.read_csv(path, header=None)

    rows = []

    for index, row in ratingsFrame.iterrows():
        rating, review = row[0], row[1]

        if isNA(review) or isNA(rating):
            continue

        if type(review) is not str:
            review = str(review)

        # Create tokens from the review text
        tokens = [Token.SOS]
        tokens += tokenize(review)
        tokens.append(Token.EOS)

        # Get a random window of the review and the corresponding target
        windows = createWindows(
            tokens,
            maxWindowSize=maxWindowSize,
            fullWindowsOnly=fullWindowsOnly
        )

        # Make a new row for the dataset
        # NOTE: In the datsets, there is no need to tokenize
        # when reading--just split on spaces.
        newRows = [[rating, ' '.join(sequence), target] for sequence, target in windows]

        rows += newRows

    data = pd.DataFrame(rows, columns=['rating', 'tokens', 'target'])

    # Take the testing data into account for the total length
    totalLength = length + int(length * testFraction)

    if len(data) < length:
        print(f"[ERROR] Available data ({len(data)}) is shorter than desired `length` ({length})!\n")

    else:
        lengthOfEach = totalLength // 5
        lengthOfEachTrain = length // 5

        # Check if any of the classes is too small
        for rating in range(5):
            lengthOfThis = len(data[data['rating'] == rating + 1])
            if lengthOfThis < lengthOfEach:
                print(f"[ERROR] Not enough {rating + 1}-star ratings to reach length (have {lengthOfThis} but need {lengthOfEach})!\n")
                # exit()

        classes = [
            data[data['rating'] == rating + 1].sample(lengthOfEach).reset_index(drop=True)
            for rating in range(5)
        ]

        trainingClasses = [ratingData[:lengthOfEachTrain] for ratingData in classes]
        testingClasses = [ratingData[lengthOfEachTrain:] for ratingData in classes]

        # Create training dataframe and shuffle
        trainingDataset = pd.concat(trainingClasses)
        trainingDataset = trainingDataset.sample(frac=1).reset_index(drop=True)

        # Create training dataframe and shuffle
        testingDataset = pd.concat(testingClasses)
        testingDataset = testingDataset.sample(frac=1).reset_index(drop=True)

        trainOutputPath = outputPath[:-4] + "-train.csv"
        trainingDataset.to_csv(trainOutputPath, index=False, header=False)

        print(f"Training dataset created (saved to '{trainOutputPath}').")

        if len(testingDataset) != 0:
            testingOutputPath = outputPath[:-4] + "-test.csv"
            testingDataset.to_csv(testingOutputPath, index=False, header=False)

            print(f"Testing dataset created (saved to '{testingOutputPath}').")




if __name__ == "__main__":
    n = 14
    # splitAmazonData(n)
    # shuffleAll(n)

    # createCategoryAndBalancedCSVs()

    createDataset(
        "data/amazon/csv/Prime_Pantry_5.csv",
        maxWindowSize= 5,
        length= 50000,      # How many training examples
        testFraction= 0.25, # Added on top of training length
        outputPath= "data/prime-pantry-50k.csv"
    )

    # createDataset(
    #     "data/amazon/csv/Prime_Pantry_5.csv",
    #     maxWindowSize=5,
    #     length=50000,
    #     outputPath="data/prime-pantry-50k.csv"
    # )