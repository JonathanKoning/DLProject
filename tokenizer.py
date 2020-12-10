import string

from nltk import word_tokenize
from LSTM import Token


def nltk_tokenize(inputString):
    return word_tokenize(inputString)


alphaNumericCharacters = string.digits + string.ascii_letters
asciiCharacters = alphaNumericCharacters + string.punctuation

tokenCharacters = set(string.ascii_letters + "'-")

numbers = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine"
}

def translateNumber(numberString):
    if len(numberString) == 1 and numberString[0] in numbers:
        return numbers[numberString[0]]

    return Token.Number


def validToken(checkString):
    return checkString != '' and all([c in tokenCharacters for c in checkString])


def isURL(checkString):
    lowerString = checkString.lower()
    if "http" in lowerString or ".com" in lowerString or ".org" in lowerString:
        return True

    if '/' in checkString and checkString.count('.') >= 2:
        return True


def stripPunctuation(inputString):
    for start, character in enumerate(inputString):
        if character not in alphaNumericCharacters:
            continue

        for end, character in enumerate(reversed(inputString)):
            stop = len(inputString) - end

            if stop <= start:
                return ''

            if character in alphaNumericCharacters:
                return inputString[start : stop]

    return ''


def stripHTML(inputString):
    htmlOpen = False
    cleanString = []
    for character in inputString:
        if character == '<':
            htmlOpen = True
            continue

        if character == '>':
            htmlOpen = False
            continue

        if not htmlOpen:
            cleanString.append(character)

    return ''.join(cleanString)


def containsLetters(inputString):
    for character in inputString:
        if character in string.ascii_letters:
            return True

    return False


def containsNumber(inputString):
    for character in inputString:
        if character in string.digits:
            return True

    return False


def splitSlashes(slashedString):
    for word in slashedString.split('/'):
        word = stripPunctuation(word)
        if validToken(word):
            yield word


wordPieces = set([
    "'m",
    "'ve",
    "n't",
    "'s",
    "'d",
    "'re",
    "'ll"
])

sentenceBoundaries = set(";.!?")


def tokenGenerator(inputString):

    # 5. Check if the token matches <num> / <date> / <url>

    # Don't split on dashes
    # Do split on slashes
    # Do split on period
    # Don't split on apostrophe's

    inputString = inputString.lower()
    inputString = inputString.replace("&nbsp;", ' ')
    inputString = stripHTML(inputString)

    tokens = nltk_tokenize(inputString)

    length = len(tokens)

    inSentence = False

    for index, entity in enumerate(tokens):
        # # Handle start / end of sentences
        # if len(entity) == 1 and entity in sentenceBoundaries and inSentence:
        #     inSentence = False
        #     yield Token.EOS

        # elif not inSentence and containsLetters(entity):
        #     inSentence = True
        #     yield Token.SOS

        # Ignore word pieces
        if entity in wordPieces:
            continue

        # Remove leading/trailing punctuation
        entity = stripPunctuation(entity)

        if validToken(entity):
            yield entity

        # Tack on word pieces following and return the combined token
        elif index + 1 < length and tokens[index + 1] in wordPieces:
            yield entity + tokens[index + 1]

        elif isURL(entity):
            yield Token.URL

        elif containsNumber(entity):
            yield translateNumber(entity)

        elif '/' in entity:
            for word in splitSlashes(entity):
                yield word

        elif len(entity) > 0:
            yield entity

    # if inSentence:
    #     yield Token.EOS



def tokenize(inputString):
    return list(tokenGenerator(inputString))


if __name__ == "__main__":
    print(stripPunctuation("(\"testing...testing)[]"))