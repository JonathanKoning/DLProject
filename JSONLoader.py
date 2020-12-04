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

asciiCharacters = set(string.printable)


def jsonToList(jsonData, fields):
        return [jsonData[f] for f in fields if f in jsonData]


def clean(dirtyString):
        asciiString = "".join(list(filter(lambda c: c in asciiCharacters, dirtyString)))

        cleanString = asciiString.replace("\\r", ' ').replace("\\n", ' ')

        return cleanString


def reddit():

    fields = ["subreddit", "subreddit_id", "score", "body"]

    with open("data/reddit/comments.json") as file:

        for line in file:
            commentData = json.loads(clean(line.strip()))
            commentDateTime = datetime.datetime.utcfromtimestamp(int(commentData["created_utc"]))
            commentDay = commentDateTime.day

            body = commentData["body"]

            if body == "[deleted]":
                continue

            yield jsonToList(commentData, fields)


def amazon(fileName):

    fields = ["overall", "reviewText"]

    def jsonToRow(jsonData):
        return [(clean(str(jsonData[f])) if f in jsonData else "") for f in fields]

    with open("data/amazon/" + fileName) as file:
        for line in file:
            reviewData = json.loads(clean(line.strip()))

            yield jsonToList(reviewData, fields)

