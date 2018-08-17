import tensorflow as tf
import numpy as np
import input

class Word2vec:

    def __init__(self, *args, **kwargs):
        self.url =  "http://mattmahoney.net/dc/"
        self.filename = "text8.zip" 
        self.vocabulary_size = 50000
        self.freature_size = 300
        self.window_size = 5

        self.__collect_data__()
        self.__create_graph__()
        
    def __collect_data__(self):
        self.filename = input.download_data(self.url, self.filename)
        self.data = input.read_data(self.filename)
        self.data, self.count, self.dictionary, self.reverse_dictionary = input.build_dataset(self.data, self.vocabulary_size)

    def __create_graph__(self):
        with tf.name_scope("graph"):
            self.X = tf.placeholder([self.vocabulary_size], dtype=tf.float32, name="X")
            self.Hidden1 = tf.Variable(tf.random_normal([self.vocabulary_size, self.freature_size]), dtype=tf.float32, name="Hidden1")
            self.feature_space = tf.matmul(self.X, self.Hidden1, name="feature")
            self.Hidden2 = tf.Variable(tf.random_normal([self.freature_size, self.vocabulary_size]), dtype=tf.float32, name="Hidden2")
            self.output = tf.matmul(self.feature_space, self.Hidden2)
            self.y = tf.nn.softmax(self.output, name="y")
            self.labels = tf.placeholder([self.vocabulary_size], dtype=tf.float32)


    def __build_loss__(self):
        with tf.name_scope("loss"):
            


    