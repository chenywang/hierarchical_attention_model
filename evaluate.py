# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import argparse
import pickle as pl

import os
import pandas as pd
import tensorflow as tf

from attention_model.hierarchical_attention_model import HierarchicalModel
from config import log_path, train_path, test_path, embedding_pickle_path
from util.data_utils import gen_batch_train_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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

    print("载入嵌入层...")
    (emb_matrix, word2index, index2word) = pl.load(open(embedding_pickle_path, "rb"))

    print("载入训练与测试数据...")
    train_size = test_size = 10000
    train_data = pd.read_csv(train_path, sep='\t')[:train_size]
    test_data = pd.read_csv(test_path, sep='\t')[:test_size]

    print("生成模型中...")
    model = HierarchicalModel(max_sentence_length, max_review_length, emb_matrix)

    saver = tf.train.Saver()

    config_proto = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    with tf.Session(config=config_proto) as sess:
        train_writer = tf.summary.FileWriter(log_path, sess.graph)
        print("载入模型中...")
        latest_cpt_file = tf.train.latest_checkpoint(log_path)
        model.restore(sess, saver, latest_cpt_file)

        print("正在评估...")
        for index, (batch_data, batch_label, review_length_list) in \
                enumerate(gen_batch_train_data(test_data, word2index, max_sentence_length, max_review_length)):
            accuracy = model.evaluate(sess, batch_data, batch_label, review_length_list)
            print("第{}个batch测试集的accuracy为{}".format(index, accuracy))
