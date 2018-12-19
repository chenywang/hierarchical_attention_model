import argparse
import os
import pickle as pl

import numpy as np
import pandas as pd
import tensorflow as tf

from process.components import sequence, attention, visualize
from process.data_utils import gen_batch_train_data
from config import config


def build_graph(
        inputs,
        revlens,
        keep_probs,
        hidden_size=50,
        atten_size=50,
        nclasses=2,
        embeddings=None
):
    # Placeholders
    print(inputs.shape)
    print(revlens.shape)

    max_rev_length = int(inputs.shape[1])
    sent_length = int(inputs.shape[2])
    print(max_rev_length, sent_length)

    _, embedding_size = embeddings.shape
    word_rnn_inputs = tf.nn.embedding_lookup(tf.convert_to_tensor(embeddings, np.float32), inputs)
    print("word rnn inputs: " + str(word_rnn_inputs))
    word_rnn_inputs_formatted = tf.reshape(word_rnn_inputs, [-1, sent_length, embedding_size])
    print('word rnn inputs formatted: ' + str(word_rnn_inputs_formatted))

    reuse_value = None

    with tf.variable_scope("word_rnn"):
        word_rnn_outputs = sequence(word_rnn_inputs_formatted, hidden_size, None)

    # now add the attention mech on words:
    # Attention mechanism at word level

    atten_inputs = tf.concat(word_rnn_outputs, 2)
    combined_hidden_size = int(atten_inputs.shape[2])

    atten_inputs = tf.nn.dropout(atten_inputs, keep_probs[0])
    with tf.variable_scope("word_atten"):
        sent_outs, alphas_words = attention(atten_inputs, atten_size)

    sent_outs_formatted = tf.reshape(sent_outs, [-1, max_rev_length, combined_hidden_size])
    print("sent outs formatted: " + str(sent_outs_formatted))
    sent_rnn_inputs_formatted = sent_outs_formatted
    print('sent rnn inputs formatted: ' + str(sent_rnn_inputs_formatted))

    with tf.variable_scope("sent_rnn"):
        sent_rnn_outputs = sequence(sent_rnn_inputs_formatted, hidden_size, revlens)

    # attention at sentence level:
    sent_atten_inputs = tf.concat(sent_rnn_outputs, 2)
    sent_atten_inputs = tf.nn.dropout(sent_atten_inputs, keep_probs[1])

    with tf.variable_scope("sent_atten"):
        rev_outs, alphas_sents = attention(sent_atten_inputs, atten_size)

    with tf.variable_scope("out_weights1", reuse=reuse_value) as out:
        weights_out = tf.get_variable(name="output_w", dtype=tf.float32, shape=[hidden_size * 2, nclasses])
        biases_out = tf.get_variable(name="output_bias", dtype=tf.float32, shape=[nclasses])
    dense = tf.matmul(rev_outs, weights_out) + biases_out
    print(dense)

    return dense, alphas_words, alphas_sents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for building the model.')
    parser.add_argument('-b', '--batch_size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('-r', '--resume', type=bool, default=False,
                        help='pick up the latest check point and resume')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='epochs for training')
    parser.add_argument('-s', '--max_sentence_length', type=int, default=70,
                        help='fix the sentence length in all reviews, 需要跟预处理保持一致')
    parser.add_argument('-r', '--max_rev_length', type=int, default=15,
                        help='fix the maximum review length, 需要跟预处理保持一致')

    args = parser.parse_args()
    train_batch_size = args.batch_size
    resume = args.resume
    epochs = args.epochs
    max_sentence_length = args.max_sentence_length
    max_rev_length = args.max_rev_length

    working_dir = config.imbd_path
    log_dir = config.log_path

    print("load embedding matrix...")
    (emb_matrix, word2index, index2word) = pl.load(open(config.embedding_pickle_path, "rb"))

    n_classes = 2
    y_ = tf.placeholder(tf.int32, shape=[None, n_classes])
    inputs = tf.placeholder(tf.int32, [None, max_rev_length, max_sentence_length])
    revlens = tf.placeholder(tf.int32, [None])
    keep_probs = tf.placeholder(tf.float32, [2])

    dense, alphas_words, alphas_sents = build_graph(inputs, revlens, keep_probs, embeddings=emb_matrix,
                                                    nclasses=n_classes)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=dense))
    with tf.variable_scope('optimizers', reuse=None):
        optimizer = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    y_predict = tf.argmax(dense, 1)
    correct_prediction = tf.equal(y_predict, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()

    num_buckets = 3

    depth = n_classes
    on_value = 1
    off_value = 0

    train_data = pd.read_csv(config.train_path, sep='\t')
    test_data = pd.read_csv(config.test_path, sep='\t')
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #         insert this snippet to restore a model:
        resume_from_epoch = -1
        if resume:
            latest_cpt_file = tf.train.latest_checkpoint(config.log_path)
            print("the code pick up from lateset checkpoint file: {}".format(latest_cpt_file))
            resume_from_epoch = int(str(latest_cpt_file).split('-')[1])
            print("it resumes from previous epoch of {}".format(resume_from_epoch))
            saver.restore(sess, latest_cpt_file)
        for epoch in range(resume_from_epoch + 1, resume_from_epoch + epochs + 1):
            avg_cost = 0.0
            print("epoch {}".format(epoch))
            # for i in range(total_batch):
            for index, (batch_data, batch_label, sentence_length_list) in \
                    enumerate(gen_batch_train_data(train_data, word2index, max_sentence_length, max_rev_length)):
                batch_label_formatted = tf.one_hot(indices=batch_label, depth=depth, on_value=on_value,
                                                   off_value=off_value, axis=-1)

                batch_labels = sess.run(batch_label_formatted)
                feed = {inputs: batch_data, revlens: sentence_length_list, y_: batch_labels, keep_probs: [0.9, 0.9]}
                _, loss, summary_in_batch_train = sess.run([optimizer, cross_entropy, summary_op], feed_dict=feed)
                print("第{}个epoch的第{}个batch的平均交叉熵为: {}".format(index, epoch, avg_cost))

            saver.save(sess, os.path.join(log_dir, "model.ckpt"), epoch, write_meta_graph=False)

        print("正在评估...")

        for index, (batch_data, batch_label, sentence_length_list) in \
                enumerate(gen_batch_train_data(test_data, word2index, max_sentence_length, max_rev_length)):
            batch_label_formatted2 = tf.one_hot(indices=batch_label, depth=depth, on_value=on_value,
                                                off_value=off_value, axis=-1)

            batch_labels2 = sess.run(batch_label_formatted2)
            feed = {inputs: batch_data, revlens: sentence_length_list, y_: batch_labels2, keep_probs: [1.0, 1.0]}
            acc = sess.run(accuracy, feed_dict=feed)
            print("第{}个batch测试集的accuracy为{}".format(index, acc))

        print("正在生成可视化h5界面")
        for i in range(100):
            visualize(sess, inputs, revlens, max_rev_length, keep_probs, index2word, alphas_words, alphas_sents, x_test,
                      y_test, y_predict, i)
