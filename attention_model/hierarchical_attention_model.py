import os

import numpy as np
import tensorflow as tf

from algorithm.implement.components import attention, sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class HierarchicalModel:
    def __init__(self, max_sentence_length, max_review_length, embeddings, hidden_size=50, attention_size=50,
                 n_classes=2):
        self.max_sentence_length = max_sentence_length
        self.max_review_length = max_review_length
        self.embeddings = embeddings
        self.embedding_dims = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_classes = n_classes
        self.summary_op = None
        self.optimizer = None
        self.on_value = 1
        self.off_value = 0
        self.accuracy = 0

        self.y_ = None
        self.inputs = None
        self.review_length_list = None
        self.keep_probabilities = None
        self.cross_entropy = None
        self.alphas_words = None
        self.alphas_sentences = None
        self.dense = None
        self.soft_max = None

        self.init_placeholders()
        self.build_model()

    def init_placeholders(self):
        self.y_ = tf.placeholder(tf.int32, shape=[None], name='y_place_holder')
        self.inputs = tf.placeholder(tf.int32, [None, self.max_review_length, self.max_sentence_length],
                                     name='x_place_holder')
        self.review_length_list = tf.placeholder(tf.int32, [None], name='review_length_place_holder')
        self.keep_probabilities = tf.placeholder(tf.float32, [2], name='keep_probabilities_place_holder')

    def build_model(self):
        # embedding层
        embedding_output = tf.nn.embedding_lookup(tf.convert_to_tensor(self.embeddings, np.float32), self.inputs)
        embedding_output = tf.reshape(embedding_output, [-1, self.max_sentence_length, self.embedding_dims])

        with tf.variable_scope("word_rnn"):
            word_rnn_outputs = sequence(embedding_output, self.hidden_size, None)

        attention_inputs = tf.concat(word_rnn_outputs, 2)
        combined_hidden_size = int(attention_inputs.shape[2])

        attention_inputs = tf.nn.dropout(attention_inputs, self.keep_probabilities[0])
        with tf.variable_scope("word_attention"):
            word_attention_output, self.alphas_words = attention(attention_inputs, self.attention_size)

        word_attention_output = tf.reshape(word_attention_output, [-1, self.max_review_length, combined_hidden_size])

        with tf.variable_scope("sentence_rnn"):
            sentence_rnn_outputs = sequence(word_attention_output, self.hidden_size, self.review_length_list)

        sentence_attention_inputs = tf.concat(sentence_rnn_outputs, 2)
        sentence_attention_inputs = tf.nn.dropout(sentence_attention_inputs, self.keep_probabilities[1])

        with tf.variable_scope("sentence_attention"):
            sentence_attention_outputs, self.alphas_sentences = attention(sentence_attention_inputs,
                                                                          self.attention_size)

        # 全连接层
        with tf.variable_scope("out_weights1"):
            weights_out = tf.get_variable(name="output_w", dtype=tf.float32,
                                          shape=[self.hidden_size * 2, self.n_classes])
            biases_out = tf.get_variable(name="output_bias", dtype=tf.float32, shape=[self.n_classes])
        self.dense = tf.matmul(sentence_attention_outputs, weights_out) + biases_out

        # 优化层
        one_hot_y = tf.one_hot(indices=self.y_, depth=self.n_classes, on_value=self.on_value, off_value=self.off_value,
                               axis=-1)
        self.soft_max = tf.nn.softmax(self.dense)
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=self.dense))
        with tf.variable_scope('optimizers', reuse=None):
            self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cross_entropy)
        y_predict = tf.argmax(self.dense, 1)
        correct_prediction = tf.equal(y_predict, tf.argmax(one_hot_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("cost", self.cross_entropy)
        tf.summary.scalar("accuracy", self.accuracy)

        self.summary_op = tf.summary.merge_all()

    def train(self, sess, x_data, y_data, sentence_length_list):
        feed = {self.inputs: x_data, self.review_length_list: sentence_length_list, self.y_: y_data,
                self.keep_probabilities: [0.9, 0.9]}
        _, loss, summary_in_batch_train = sess.run([self.optimizer, self.cross_entropy, self.summary_op],
                                                   feed_dict=feed)
        return loss

    def evaluate(self, sess, x_data, y_data, sentence_length_list):
        feed = {self.inputs: x_data, self.review_length_list: sentence_length_list, self.y_: y_data,
                self.keep_probabilities: [1.0, 1.0]}
        accuracy = sess.run([self.accuracy], feed_dict=feed)
        return accuracy

    def predict(self, sess, x_data, review_length_list):
        feed = {self.inputs: x_data, self.review_length_list: review_length_list,
                self.keep_probabilities: [1.0, 1.0]}
        soft_max, alphas_words, alphas_sentences = sess.run([self.soft_max, self.alphas_words, self.alphas_sentences],
                                                            feed_dict=feed)
        return soft_max, alphas_words, alphas_sentences

    def predict_single(self, sess, x, review_length):
        return self.predict(sess, [x], [review_length])

    @staticmethod
    def save(sess, saver, path, global_step=None):
        save_path = saver.save(sess, save_path=path, global_step=global_step, write_meta_graph=False)
        print('模型被保存在{}'.format(save_path))

    @staticmethod
    def restore(sess, saver, model_path):
        saver.restore(sess, model_path)
        print("模型载入完成")
