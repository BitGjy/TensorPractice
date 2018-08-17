import tensorflow as tf
import numpy as np
import input

class Word2vec:

    def __init__(self, *args, **kwargs):
        self.url =  "http://mattmahoney.net/dc/"
        self.filename = "text8.zip" 
        self.vocabulary_size = 50000
        
    def __collect_data__(self):
        self.filename = input.download_data(self.url, self.filename)
        self.data = input.read_data(self.filename)
        self.data, self.count, self.dictionary, self.reverse_dictionary = input.build_dataset(self.data, self.vocabulary_size)

    def __create_graph__(self):
        

    