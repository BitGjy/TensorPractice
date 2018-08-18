import tensorflow as tf
import numpy as np
import input

class Word2vec:

    def __init__(self, *args, **kwargs):
        self.url =  "http://mattmahoney.net/dc/"
        self.filename = "text8.zip" 
        self.vocabulary_size = 50000
        self.feature_size = 300
        self.window_size = 5
        self.batch_size = 128
        self.num_sampled = 500
        self.epoch = 10
        self.train_sents_num = 10000

        self.train_loss_records = []
        self.train_loss_k10 = []

        self.summary_writer = tf.summary.FileWriter(logdir="./logs")
        self.summary_str = "./logs"


        self.__collect_data__()
        self.__create_graph__()
        self.__build_train__()
        self.__init_values__()

      
    def __collect_data__(self):
        self.filename = input.download_data(self.url, self.filename)
        self.data = input.read_data(self.filename)
        self.data, self.count, self.dictionary, self.reverse_dictionary = input.build_dataset(self.data, self.vocabulary_size)
    

    def __create_graph__(self):
        with tf.name_scope("graph"):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.train_inputs = tf.placeholder(tf.float32, shape=[self.batch_size])
                self.train_labels = tf.placeholder(tf.float32, shape=[self.batch_size, 1])

                self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.feature_size], -1.0, 1.0))
            
                self.nce_weights = tf.Variable(tf.truncated_normal(self.vocabulary_size, self.feature_size, stddev=1.0/np.sqrt(self.feature_size)))
                self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

                self.loss = tf.reduce_mean(tf.nn.nce_loss(
                    weights=self.nce_weights,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=self.embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocabulary_size
                ))

                tf.summary.scalar('loss', self.loss)
            
    def __build_train__(self):
        with tf.name_scope("train"):
            self.train = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
            self.test_word_id = tf.placeholder(tf.int32, shape=[None])
            
            vec_l2_model = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            avg_l2_model = tf.reduce_mean(vec_l2_model)

            tf.summary.scalar('avg_l2_model', avg_l2_model)

            self.normed_embeddings = self.embeddings / vec_l2_model
            # self.embedding_dict = norm_vec # 对embedding向量正则化
            test_embed = tf.nn.embedding_lookup(self.normed_embeddings, self.test_word_id)
            self.similarity = tf.matmul(test_embed, self.normed_embeddings, transpose_b=True)

            # 变量初始化操作
            self.init = tf.global_variables_initializer()

            #with tf.Session() as sess:
            #    sess.run(self.init)

    def __init_values__(self):
        tf.Session().run(self.init)
                
    def train_by_sequence(self, input_sequence=[]):
        sentence_num = input_sequence.__len__()

        batch_inputs = []
        batch_labels = []

        for sent in input_sequence:
            for i in range(sent.__len__()):
                start = max(0, i-mod.window_size)
                end = min(sent.__len__(), i+mod.window_size+1)

                for index in range(start, end):
                    if index == i:
                        continue
                    else:
                        input_id = mod.dictionary[sent[i]]
                        label_id = mod.dictionary[sent[index]]

                        if not (input_id and label_id):
                            continue

                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)

                        if len(batch_inputs) == 0:
                            return
                    
        batch_inputs = np.array(batch_inputs, dtype=np.int32)
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])

        feed_dict = {
            self.train_inputs:batch_inputs,
            self.train_labels:batch_labels
        }

        with tf.Session() as sess:
            loss_val = sess.run(self.loss, feed_dict=feed_dict)

            self.train_loss_records.append(loss_val)
            self.train_loss_k10 = np.mean(self.train_loss_records)
            if self.train_sents_num % 1000 == 0 :
                self.summary_writer.add_summary(self.summary_str,self.train_sents_num)
                print("{a} sentences dealed, loss: {b}".format(a=self.train_sents_num,b=self.train_loss_k10))