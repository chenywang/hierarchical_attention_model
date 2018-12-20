# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import argparse
import os
import pickle as pl

import pandas as pd
import tensorflow as tf

from attention_model.hierarchical_attention_model import HierarchicalModel
from config import config
from util.data_utils import gen_batch_train_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for building the model.')
    parser.add_argument('-b', '--batch_size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('-r', '--resume', type=bool, default=False,
                        help='pick up the latest check point and resume')
    parser.add_argument('-m', '--max_sentence_length', type=int, default=70,
                        help='fix the sentence length in all reviews, 需要跟预处理保持一致')
    parser.add_argument('-M', '--max_rev_length', type=int, default=15,
                        help='fix the maximum review length, 需要跟预处理保持一致')

    # 设置基本参数
    args = parser.parse_args()
    train_batch_size = args.batch_size
    resume = args.resume
    max_sentence_length = args.max_sentence_length
    max_review_length = args.max_rev_length
    log_dir = config.log_path
    on_value = 1
    off_value = 0
    depth = n_classes = 2

    print("载入嵌入层...")
    (emb_matrix, word2index, index2word) = pl.load(open(config.embedding_pickle_path, "rb"))

    print("载入训练与测试数据...")
    train_size = test_size = 1000
    train_data = pd.read_csv(config.train_path, sep='\t')[:train_size]
    test_data = pd.read_csv(config.test_path, sep='\t')[:test_size]

    print("生成模型中...")
    model = HierarchicalModel(max_sentence_length, max_review_length, emb_matrix)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("载入模型中...")
        latest_cpt_file = tf.train.latest_checkpoint(config.log_path)
        model.restore(sess, saver, latest_cpt_file)

        print("正在评估...")
        for index, (batch_data, batch_label, review_length_list) in \
                enumerate(gen_batch_train_data(test_data, word2index, max_sentence_length, max_review_length)):
            accuracy = model.evaluate(sess, batch_data, batch_label, review_length_list)
            print("第{}个batch测试集的accuracy为{}".format(index, accuracy))
