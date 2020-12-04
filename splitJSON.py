import datetime
import json
import string


fields = ["subreddit", "subreddit_id", "score", "body"]

asciiCharacters = set(string.printable)

def clean(dirtyString):
    asciiString = "".join(list(filter(lambda c: c in asciiCharacters, dirtyString)))

    specialCharacters = set("\r\t\n")
    cleanString = "".join(list(map(lambda c: ' ' if c in specialCharacters else c, asciiString)))

    return cleanString


def jsonToRow(jsonData):
    return "\t".join([(clean(str(jsonData[f])) if f in jsonData else "") for f in fields])


with open("Data/comments") as file:
    currentDay = 1
    lines = []

    for line in file:
        commentData = json.loads(line.strip())
        commentDateTime = datetime.datetime.utcfromtimestamp(int(commentData["created_utc"]))
        commentDay = commentDateTime.day

        body = commentData["body"]

        if body == "[deleted]":
            continue

        if commentDay < currentDay:
            print("Date error: ")
            print(commentData)
            print("^^^")

        if commentDay > currentDay:
            fileName = "./Data/reddit/" + str(currentDay) + ".tsv"

            print(f"Finished day {currentDay}. Saved to '{fileName}'.")

            with open(fileName, 'w') as outputFile:
                outputFile.writelines(lines)

            lines = []
            currentDay += 1

            if commentDay != currentDay:
                print("Date change-over error: ")
                print(commentDay, currentDay, commentData)
                print("^^^")


        lines.append(jsonToRow(commentData) + "\n")

