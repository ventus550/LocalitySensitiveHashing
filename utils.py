import itertools
import numpy as np
from string import ascii_lowercase
from random import randint, choice
from collections import Counter

vocabulary = list(itertools.permutations(ascii_lowercase, 2))

def shingle(text: str, k: int = 2):
    return set( tuple(text[i:i+k]) for i in range(len(text) - k+1) )

def vectorize(str: str, shingles = True):
    if shingles:
        shingles = shingle(str)
        return np.array([t in shingles for t in vocabulary], dtype=int)
    counts = Counter(str)
    return np.array([counts[letter] for letter in ascii_lowercase])

def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Deletion
                                   dp[i][j - 1],      # Insertion
                                   dp[i - 1][j - 1])  # Substitution
    return dp[m][n]

def noise(word: str):
    i = randint(0, len(word))
    return word[:i] + choice(ascii_lowercase) + word[i+1:]