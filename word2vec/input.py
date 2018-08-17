import argparse as arp
import tensorflow as tf
import numpy as np
import os
from urllib.request import urlretrieve
import zipfile
import collections


def download_data(url, filename):
    if not os.path.exists(filename):
        filename, _ =  urlretrieve(url=url+filename, filename=filename)
        statinfo = os.stat(filename)
        '''
        if statinfo.st_size == expected_bytes:
            print("Found and verified ", filename)
        else:
            print(statinfo.st_size)
        raise Exception(
            "Failed to verfy" + filename + "Can you get to it with browser?")
        '''
    return filename

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

'''
file= 'text8.zip'
expected = 31344016

filename = download_data(file, expected)

words = read_data(file)
print(np.shape(words))
print("Data_Size: ", len(words))
'''

#vocabulary_size = 50000

def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
'''
data, count, dictionary, reverse_dictionary = build_dataset(words)


del words
print("Size: ", len(count))
print ('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
'''
