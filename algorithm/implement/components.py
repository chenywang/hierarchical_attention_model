import os

import tensorflow as tf

from config import config
from util.tool import redistribute


def get_sentence(vocabulary_inv, sen_index):
    return ' '.join([vocabulary_inv[index] for index in sen_index])


def sequence(rnn_inputs, hidden_size, seq_lens):
    cell_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print('build fw cell: ' + str(cell_fw))
    cell_bw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print('build bw cell: ' + str(cell_bw))
    rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                               cell_bw,
                                                               inputs=rnn_inputs,
                                                               sequence_length=seq_lens,
                                                               dtype=tf.float32
                                                               )
    print('rnn outputs: ' + str(rnn_outputs))
    print('final state: ' + str(final_state))

    return rnn_outputs


def attention(attention_inputs, attention_size):
    """
    attention mechanism uses Ilya Ivanov's implementation(https://github.com/ilivans/tf-rnn-attention)
    :param attention_inputs:
    :param attention_size:
    :return:
    """
    max_time = int(attention_inputs.shape[1])
    combined_hidden_size = int(attention_inputs.shape[2])
    W_omega = tf.Variable(tf.random_normal([combined_hidden_size, attention_size], stddev=0.1, dtype=tf.float32))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1, dtype=tf.float32))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1, dtype=tf.float32))

    v = tf.tanh(
        tf.matmul(tf.reshape(attention_inputs, [-1, combined_hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    # u_omega is the summarizing question vector
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, max_time])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    atten_outs = tf.reduce_sum(attention_inputs * tf.reshape(alphas, [-1, max_time, 1]), 1)
    return atten_outs, alphas


def visualize(review, alphas_words_list, alphas_sentences_list, index, y, probability):
    sentence_list = list(filter(lambda x: len(x) != 0, review.split('.')))
    review_length = len(sentence_list)

    h5_path = os.path.join(config.visualization_path, "visualize_{}.html".format(index))
    with open(h5_path, "w") as h5_file:
        actual_label = '消极' if y == 1 else '积极'
        predict_label = '消极' if probability[1] >= 0.5 else '积极'
        h5_file.write("<head><meta charset='utf-8'></head>")
        h5_file.write("该评论实际：{}，模型预测：{}，消极概率：{})<br>"
                      .format(actual_label, predict_label, probability[1]))
        # 去除空句子的分担的比例
        alphas_sentences_list = redistribute(alphas_sentences_list[:review_length])
        for sentence, alphas_words, alphas_sentence in zip(sentence_list, alphas_words_list, alphas_sentences_list):
            h5_file.write(
                '<font style="background: rgba(255, 0, 0, %f)">&nbsp&nbsp&nbsp&nbsp&nbsp</font>' % alphas_sentence)
            # 去除空字的分担的比例
            words = sentence.split(' ')
            alphas_words = redistribute(alphas_words[:len(words)])
            for word, alphas_word in zip(words, alphas_words):
                h5_file.write('<font style="background: rgba(255, 255, 0, %f)">%s </font>' % (alphas_word, word))
            h5_file.write('<br>')
