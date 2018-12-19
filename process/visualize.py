# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import argparse
import os
import pickle as pl

import pandas as pd
import tensorflow as tf

from config import config
from process.data_utils import gen_batch_train_data
from process.model import HierarchicalModel


def visualize(sentence_list, alphas_words, alphas_sentences):
    size = batch_x.shape[0]
    for i in range(size):
        h5_path = os.path.join(config.visualization_path, "visualize_{}.html".format(i))
        with open(h5_path, "w") as h5_file:
            actual_label = '消极' if batch_y[i] == 1 else '积极'
            predict_label = '消极' if probability_list[i][1] >= 0.5 else '积极'
            h5_file.write("该评论实际是{}评论，模型预测是{}评论个.({}的消极概率)"
                          .format(actual_label, predict_label, probability_list[i][1]))
            h5_file.write('<font style="background: rgba(255, 0, 0, %f)">&nbsp&nbsp&nbsp&nbsp&nbsp</font>' % (alpha))
            for sentence in paragraph_list[i]:
                h5_file.write(
                    '<font style="background: rgba(255, 255, 0, %f)">%s </font>' % (alphas_words[i], sentence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for building the model.')
    parser.add_argument('-b', '--batch_size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('-r', '--resume', type=bool, default=False,
                        help='pick up the latest check point and resume')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='epochs for training')
    parser.add_argument('-m', '--max_sentence_length', type=int, default=70,
                        help='fix the sentence length in all reviews, 需要跟预处理保持一致')
    parser.add_argument('-M', '--max_rev_length', type=int, default=15,
                        help='fix the maximum review length, 需要跟预处理保持一致')

    # 设置基本参数
    args = parser.parse_args()
    batch_size = args.batch_size
    resume = args.resume
    epochs = args.epochs
    max_sentence_length = args.max_sentence_length
    max_review_length = args.max_rev_length
    log_dir = config.log_path
    on_value = 1
    off_value = 0
    depth = n_classes = 2

    print("载入嵌入层...")
    (emb_matrix, word2index, index2word) = pl.load(open(config.embedding_pickle_path, "rb"))

    print("生成模型中...")
    model = HierarchicalModel(max_sentence_length, max_review_length, emb_matrix)

    saver = tf.train.Saver()

    print("载入需要可视化的数据...")
    test_size = 100
    test_data = pd.read_csv(config.test_path, sep='\t')[:test_size]

    review_list = list()
    alphas_words_list = list()
    alphas_sentences_list = list()
    review_probability_list = list()
    y_list = list()

    with tf.Session() as sess:
        print("载入模型中...")
        latest_cpt_file = tf.train.latest_checkpoint(config.log_path)
        model.restore(sess, saver, latest_cpt_file)

        for batch_data, batch_label, sentence_length_list in gen_batch_train_data(test_data, word2index,
                                                                                  max_sentence_length,
                                                                                  max_review_length,
                                                                                  batch_size=batch_size):

            probability_list, alphas_words, alphas_sentences = model.predict(sess, batch_data, batch_label,
                                                                             sentence_length_list)
