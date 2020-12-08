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


import glob, json, os, string

import pandas as pd
import numpy as np

from math import ceil
from random import shuffle, random, choice
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from LSTM import isNA
from embedding import tokenize, Token

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


def shuffled(df):
    return df.sample(frac=1).reset_index(drop=True)


# Converts ratings to ints
def ratingsAsIntegers(df):
    return df.astype({"rating": int})


def save(df, path):
    df.to_csv(path, index=False, header=False)


def loadFromJSON(path, verbose=False):
    progressMeter = "|/-\\"
    progressIndex = 0
    # What are the JSON fields we want
    fields = ["overall", "reviewText"]
    # What do we want to call those columns in pandas
    columns = ["rating", "review"]

    reviews = []

    # Converts amazon review object to list using `fields`
    def jsonToRow(jsonData):
        return [(clean(str(jsonData[f])) if f in jsonData else "") for f in fields]

    with open(path) as file:
        for number, line in enumerate(file):
            reviews.append(jsonToList(json.loads(line.strip()), fields))

            if verbose and number % 1000 == 0:
                print(f" ({progressMeter[progressIndex]}) line {number} ... ", end='\r')

                if number % 2000 == 0:
                    progressIndex = (progressIndex + 1) % 4

    # Convert the data in a Dataframe with ratings as integers
    reviewDataFrame = ratingsAsIntegers(pd.DataFrame(reviews, columns=columns))

    return reviewDataFrame


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


# Visualize and output the occurences
def histogramRatings(path, datasetFile=False):
    names = ['rating', 'tokens', 'target'] if datasetFile else ['rating', 'review']
    df = pd.read_csv(path, header=None, index_col=None, names=names)

    print(Counter(df["rating"].to_numpy()))


def balanceRatings(dataFrame, trainLength=None, testLength=None, testSplit=None):

    dataFrame = shuffled(dataFrame)

    if trainLength is not None and testLength is not None:
        if testLength < 5:
            print(f"[WARNING] Test length ({testLength}) is too short to generate balanced data.")

        lengthOfEach = (trainLength + testLength) // 5

        # Check if any of the classes is too small
        for rating in range(5):
            lengthOfThis = len(dataFrame[dataFrame['rating'] == rating + 1])
            if lengthOfThis < lengthOfEach:
                print(f"[ERROR] Not enough {rating + 1}-star ratings to reach length (have {lengthOfThis} but need {lengthOfEach})!\n")
                return

    else:
        # Find the length of the minority rating
        lengthOfEach = min([len(dataFrame[dataFrame['rating'] == rating + 1]) for rating in range(5)])

    classes = [
        dataFrame[dataFrame['rating'] == rating + 1].sample(lengthOfEach).reset_index(drop=True)
        for rating in range(5)
    ]

    lengthOfEachTrain = None

    # If the caller specified the length they want
    if trainLength is not None and testLength is not None:
        lengthOfEachTrain = trainLength // 5

    # If the user just wants a fractional split
    elif testSplit is not None:
        lengthOfEachTest  = ceil(lengthOfEach * testSplit)
        lengthOfEachTrain = lengthOfEach - lengthOfEachTest

    testingClasses = []

    # If we need to do a training split
    if lengthOfEachTrain is not None:
        trainingClasses = [ratingData[:lengthOfEachTrain] for ratingData in classes]
        testingClasses  = [ratingData[lengthOfEachTrain:] for ratingData in classes]
    else:
        trainingClasses = classes

    # Create training dataframe and shuffle
    trainingDataset = shuffled(pd.concat(trainingClasses))

    # Create training dataframe and shuffle
    testingDataset = shuffled(pd.concat(testingClasses))

    return trainingDataset, testingDataset


def createTestAndTrainCSV(testSplit):
    """Converts all .json files into .csv's while also
    splitting into train and test files. The test split is
    guaranteed to be greater than or equal to the fraction
    specified by `testSplit`.

    Args:
        testSplit (int): What fraction of data is held out for
                         testing
    """

    jsonDirectory = os.path.join(currentDirectory(), "data/json/")
    csvDirectory  = os.path.join(currentDirectory(), "data/csv/")

    for path in glob.glob(jsonDirectory + "*.json"):

        print(f"Converting '{path}' ...")

        name = os.path.basename(path)[:-7].replace('_', '-').lower()

        trainFilePath = csvDirectory + name + "-train.csv"
        testFilePath =  csvDirectory + name + "-test.csv"

        reviewData = loadFromJSON(path, verbose=True)

        train, test = balanceRatings(reviewData, testSplit=testSplit)

        save(train, trainFilePath)
        save(test, testFilePath)


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
        trainLength = length
        testLength = totalLength - trainLength

        trainingDataset, testingDataset = balanceRatings(data, trainLength, testLength)

        trainOutputPath = outputPath[:-4] + "-train.csv"
        trainingDataset.to_csv(trainOutputPath, index=False, header=False)

        print(f"Training dataset created (saved to '{trainOutputPath}').")

        if len(testingDataset) != 0:
            testingOutputPath = outputPath[:-4] + "-test.csv"
            testingDataset.to_csv(testingOutputPath, index=False, header=False)

            print(f"Testing dataset created (saved to '{testingOutputPath}').")


if __name__ == "__main__":

    # Create .csv's for the embeddings learner to consume
    createTestAndTrainCSV(0.15)

    # Create LSTM training set
    createDataset(
        "data/csv/prime-pantry-train.csv",
        maxWindowSize= 5,
        length= 100000,
        outputPath= "data/prime-pantry-100k-train.csv"
    )

    # Create LSTM testing set
    createDataset(
        "data/csv/prime-pantry-test.csv",
        maxWindowSize= 5,
        length= 25000,
        outputPath= "data/prime-pantry-25k-test.csv"
    )

    # Create smaller LSTM testing set
    createDataset(
        "data/csv/prime-pantry-test.csv",
        maxWindowSize= 5,
        length= 10000,
        outputPath= "data/prime-pantry-10k-test.csv"
    )
