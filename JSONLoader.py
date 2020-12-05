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

    fields = ["overall", "reviewText"]

    reviews = []

    def jsonToRow(jsonData):
        return [(clean(str(jsonData[f])) if f in jsonData else "") for f in fields]

    with open(path) as file:
        for line in file:
            reviews.append(jsonToList(json.loads(line.strip()), fields))

    reviewDataFrame = pd.DataFrame(reviews, columns=["rating", "review"])

    outPath = "data/amazon/csv/" + os.path.basename(path)[:-5] + ".csv"
    reviewDataFrame.to_csv(outPath)

# path = "data/amazon/Home_and_Kitchen_5.json"
# amazonDataFrame = pd.read_json(path, lines=True)
# amazonDataFrame.to_csv("data/amazon/Home_and_Kitchen_5.csv")

# print(amazonDataFrame)
for path in glob.glob("data/amazon/*.json"):
    print(path, "...")
    amazon(path)
