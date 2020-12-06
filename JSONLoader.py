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

from random import shuffle

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

    directory = "data/amazon/"

    path = "data/amazon/Prime_Pantry_5.json"

    # createCSVFiles(n, directory + "csv/", columns)

    # jsonDataPaths = os.path.join(currentDirectory(), directory + "*.json")

    # for pathNumber, path in enumerate(glob.glob(jsonDataPaths)):
        # print(f"[{pathNumber + 1}/28]", path, "...")

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
    outPath = os.path.join(currentDirectory(), directory + "csv/" + name)
    reviewDataFrame.to_csv(outPath)

    # # Randomize and split the reviews amongst `n` buckets to output
    # shuffle(reviews)
    # outputs = np.array_split(np.array(reviews), n)

    # # Create `n` data frames for each output
    # outputFrames = [pd.DataFrame(outputs[i]) for i in range(n)]

    # # Append the dataframes
    # for outputIndex in range(n):
    #     name = str(outputIndex) + ".csv"
    #     outPath = os.path.join(currentDirectory(), "data/amazon/csv/" + name)

    #     # Append this frame at the bottom of the file
    #     outputFrames[outputIndex].to_csv(outPath, index=False, header=None, mode='a')


def shuffleCSV(path):
    df = pd.read_csv(path)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    # Over-write the original file
    shuffled_df.to_csv(path, index=False, header=None, mode='w')


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

    # Go ahead and save this category version
    name = os.path.basename(path)[:-5] + ".csv"
    outPath = os.path.join(currentDirectory(), "data/amazon/csv/" + name)
    reviewDataFrame.to_csv(outPath, index=False, header=False)


def historgramRatings(path):



if __name__ == "__main__":
    n = 14
    # splitAmazonData(n)
    # shuffleAll(n)

    createCategoryCSV("data/amazon/Luxury_Beauty_5.json")
    shuffleCSV("data/amazon/csv/Luxury_Beauty_5.csv")
