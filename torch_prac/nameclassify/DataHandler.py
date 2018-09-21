from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('./data/names/*.txt'))

import unicodedata, string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

#print(unicode2Ascii('Ślusàrski'))


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode2Ascii(line) for line in lines]

def ReadData(path):

    category_lines = {}
    all_categories = []

    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)

    return category_lines, all_categories, n_categories