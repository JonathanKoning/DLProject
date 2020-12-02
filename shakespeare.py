import string

from typing import Union, Iterable
from collections import Counter

from vocab import *


def shakespeareCorpus():

    def loadShakespeareText():
        cleanedLines = []

        with open("shakespeare.txt") as shakespeare:

            for line in shakespeare:
                stored = line.rstrip()

                if line == "" or line[0] != ' ':
                    continue

                if '[' in line or ']' in line:
                    continue

                if "Enter" in line or "Exeunt" in line or "Exit" in line:
                    continue

                if len(line) > 2 and line[:4] != "    ":
                    line = line[line.find('.')+2:]

                line = line.strip()

                if line == "":
                    continue

                cleanedLines.append(line)

        return cleanedLines


    def processLine(line):
        newLine = []

        for character in line:
            if character in string.punctuation:
                newLine.append(' ')

            newLine.append(character.lower())

        return ''.join(newLine)

    cleanedLines = loadShakespeareText()
    cleanedLines = list(map(processLine, cleanedLines))

    return ' '.join(cleanedLines)
