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

from random import *
import csv

asciiCharacters = set(string.printable)


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


def amazon(fileName):

    n = 14

    fields = ["overall", "reviewText"]

    reviews = []

    def jsonToRow(jsonData):
        return [(clean(str(jsonData[f])) if f in jsonData else "") for f in fields]

    with open(fileName) as file:
        for line in file:
            reviews.append(jsonToList(json.loads(line.strip()), fields))

    reviewDataFrame = pd.DataFrame(reviews, columns=["rating", "review"])

    dirname = os.path.dirname(__file__)
    newname = os.path.basename(fileName)[:-5] + ".csv"
    outPath = os.path.join(dirname, 'data/amazon/csv/'+newname)
    #outPath = "./data/amazon/csv/" + os.path.basename(fileName)[:-5] + ".csv"
    outDir = os.path.join(dirname,"data/amazon/csv/")
    
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    reviewDataFrame.to_csv(outPath)


    for i in range(1,n+1):
        path = outDir + str(i) + ".csv"
        with open(path, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(fields)
    
    print(reviewDataFrame.iloc[1])
    for i in range(reviewDataFrame.size):
        num = randint(1, n)
        path = outDir + str(num) + ".csv"
        #print(reviewDataFrame.iloc[i])
        #with open(path, 'a') as csvfile:
            #csvwriter = csv.writer(csvfile)
            #csvwriter.writerow(reviewDataFrame.iloc[i])
        
        

# path = "data/amazon/Home_and_Kitchen_5.json"
# amazonDataFrame = pd.read_json(path, lines=True)
# amazonDataFrame.to_csv("data/amazon/Home_and_Kitchen_5.csv")

# print(amazonDataFrame)
print("HI")
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/amazon/*.json')
for path in glob.glob(filename):
    print(path)
    if path.endswith(".json"):
        print(path, "...")
        amazon(path)
