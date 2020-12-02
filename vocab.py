import string
import numpy as np

from typing import Union, Iterable
from collections import Counter


def oneHotify(scalars, size):
    """Turns a numpy array of scalars (e.g., [1, 3, None, 6]) into a one hot matrix. Ignores `None`, NaN, and negative scalars."""
    length  = len(scalars)

    indices = []
    hotOnes = []
    for index, scalar in enumerate(scalars):
        if scalar is not None and not np.isnan(scalar) and scalar >= 0:
            indices.append(index)
            hotOnes.append(scalar)

    # Create the one-hot array
    oneHotArray = np.zeros((length, size))
    # Use array indexing to match the scalars to their ordinal position
    oneHotArray[np.array(indices, dtype=int), np.array(hotOnes, dtype=int)] = 1

    return oneHotArray


# Hides the intermediate variables by being a function
def stopWords():
    # The various sources of stop words
    pronouns = {"i", "me", "us", "you", "she", "her", "he", "him", "it", "we", "us", "they", "them", "this", "these"}
    # Source: https://en.wikipedia.org/wiki/English_personal_pronouns
    copulae = {"be", "is", "am", "are", "being", "was", "were", "been"}
    # Source: https://en.wikipedia.org/wiki/Copula_(linguistics)#English
    conjunctions = {"for", "and", "nor", "but", "or", "yet", "so", "that", "which", "because", "as", "since", "though", "while", "whereas"}
    # Source: https://en.wikipedia.org/wiki/Conjunction_(grammar)
    others = {"a", "the"}

    return others.union(pronouns).union(copulae).union(conjunctions)

# Hides the function
stopWords = stopWords()


def tokenize(corpus: str, ignoreCharacters: str = "", intraWordPunctuation : str = "'") -> [str]:
    """ Turns a text corpus into a list of tokens.

    Args:
        corpus (str): Textual data
        ignore (str, optional): Characters to ignore (only matters for thing in `string.punctuation`). Defaults to "".
        intraWordPunctuation (str, optional): Punctuation to allow inside/prefixed/postfixed to words. Defaults to "'".

    Returns:
        ([str]): A list of tokens.

    Example:
    ```
    >>> tokenize("I'm a teapot")
    ["i'm", "a", "teapot"]
    ```
    """
    tokens = []
    buffer = []

    if type(ignoreCharacters) is str:
        ignoreCharacters = set(list(ignoreCharacters))

    for i, character in enumerate(corpus):
        if character in ignoreCharacters:
            continue

        isInterWordPunctuation = character in string.punctuation and character not in intraWordPunctuation

        # End the current word when there is a space, newline, or punctuation not allowed in words
        if character == '\n' or character == ' ' or isInterWordPunctuation:
            tokens.append(''.join(buffer))
            buffer = []

            # Treat punctuation as a single token
            if isInterWordPunctuation:
                tokens.append(character)

            continue

        # Treat punctuation in intraWordPunctuation as part of the word
        if character in string.punctuation:
            buffer.append(character)
            continue

        if character in string.ascii_uppercase:
            buffer.append(character.lower())
            continue

        if character in string.ascii_lowercase:
            buffer.append(character)
            continue

    return tokens


# Enumeration but I'm lazy so no `enumeration` parent
class Token():
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"


class Vocabulary():
    """
    Maps words to indices and vice versa with standard index notation.
    Returns `None` when a word cannot be found, and throws KeyError
    when an index is out of range.

    New words can be added with `addWord()`.

    Example:
    ```
    >>> vocabulary = Vocabulary(["i'm", "a", "teapot"])
    >>> print(vocabulary["a"])
    5
    >>> print(vocabulary["supercalifragilisticexpialidocious"])
    None
    >>> print(vocabulary[6])
    "teapot"
    >>> print(vocabulary.count) # Includes the special tokens (e.g. <sos>)
    6
    ```
    """

    def __init__(self,
        tokens: Iterable[str],
        min_freq: Union[int, None] = 1,
        specialTokens: Union[Iterable[str], None] = [Token.SOS, Token.EOS, Token.UNK]
    ):

        # Maps word to index
        self.indices = {k:v for v,k in enumerate(specialTokens)}
        # Maps index to word
        self.words = {v: k for k,v in self.indices.items()}
        # Number of items in the vocabulary
        self.count = len(specialTokens)

        # Add all tokens to the vocabulary and check for duplicates
        if min_freq is None:
            for token in tokens:
                self.addWord(word)

        # Only add tokens to the vocabulary with sufficient frequency
        else:
            tokenCounter = Counter(tokens) # Maps word to occurrences in corpus

            for token, occurrences in tokenCounter.items():
                if occurrences < min_freq:
                    continue

                self.addWord(token)


    def addWord(self, word: str):
        """Adds a word to the vocabulary if not already in vocabulary. Ignores ''.

        Args:
            word (str): New vocabulary word
        """
        if word == '':
            return

        if word in self.indices:
            return

        index = self.count
        self.indices[word] = index
        self.words[index]  = word
        self.count += 1


    def matricize(self, sentence):
        """Converts a sentence into a one-hot matrix of size (word count, vocabulary size)"""

        # NOTE: `tokenize()` takes extra parameters, but the defaults are used by this function.
        # TODO ? : Find a solution to above problem?

        return oneHotify([self[token] for token in tokenize(sentence)], self.count)


    def __len__(self):
        pass


    def __getitem__(self, key: Union[str, int, slice]):

        if type(key) is str:
            query = key.lower()

            if query not in self.indices:
                # raise KeyError(f"'{key}' not found in vocabulary")
                return None

            return self.indices[query]

        elif type(key) is int:
            if key not in self.words:
                raise IndexError(f"{key} outside vocabulary range")

            return self.words[key]

        elif type(key) is slice:
            start = key.start if key.start is not None else 0
            stop  = key.stop  if key.stop is not None  else self.count
            step  = key.step  if key.step is not None  else 1

            return {k: self.words[k] for k in range(start, stop, step)}

        else:
            raise KeyError(f"Vocabulary must be indexed via int or str, not '{type(key)}'")


