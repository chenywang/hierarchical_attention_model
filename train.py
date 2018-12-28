# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import argparse
import ast
import os
import pickle as pl
import time

import pandas as pd
import tensorflow as tf

from attention_model.hierarchical_attention_model import HierarchicalModel
from config import log_path, train_path, test_path, embedding_pickle_path
from util.data_utils import gen_batch_train_data

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for building the model.')
    parser.add_argument('-b', '--batch_size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('-r', '--retrain', type=ast.literal_eval, default=True,
                        help='pick up the latest check point and resume')
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help='epochs for training')

    # 设置基本参数
    args = parser.parse_args()
    train_batch_size = args.batch_size
    epochs = args.epoch
    max_sentence_length = 70
    max_review_length = 15

    print("载入嵌入层...")
    (emb_matrix, word2index, index2word) = pl.load(open(embedding_pickle_path, "rb"))

    print("载入训练与测试数据...")
    train_size = test_size = 500000
    train_data = pd.read_csv(train_path, sep='\t')[:train_size]
    test_data = pd.read_csv(test_path, sep='\t')[:test_size]

    print("生成模型中...")
    model = HierarchicalModel(max_sentence_length, max_review_length, emb_matrix)

    saver = tf.train.Saver()

    config_proto = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    with tf.Session(config=config_proto) as sess:
        train_writer = tf.summary.FileWriter(log_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #         insert this snippet to restore a model:
        resume_from_epoch = -1
        print("正在训练...")
        if not args.retrain:
            latest_cpt_file = tf.train.latest_checkpoint(log_path)
            print("the code pick up from lateset checkpoint file: {}".format(latest_cpt_file))
            model.restore(sess, saver, latest_cpt_file)
        for epoch in range(resume_from_epoch + 1, resume_from_epoch + epochs + 1):
            for index, (batch_data, batch_label, review_length_list) in \
                    enumerate(gen_batch_train_data(train_data, word2index, max_sentence_length, max_review_length,
                                                   batch_size=train_batch_size)):
                t1 = time.time()
                loss = model.train(sess, batch_data, batch_label, review_length_list)
                print("第{}个epoch的第{}个batch的交叉熵为: {},用时为{},batch size为{}".format(epoch,
                                                                                index,
                                                                                loss,
                                                                                time.time() - t1,
                                                                                train_batch_size))

            model.save(sess, saver, os.path.join(log_path, "model.ckpt"), epoch)

            print("正在评估...")
            for index, (batch_data, batch_label, review_length_list) in \
                    enumerate(gen_batch_train_data(test_data[:10000], word2index, max_sentence_length, max_review_length,
                                                   batch_size=train_batch_size)):
                accuracy = model.evaluate(sess, batch_data, batch_label, review_length_list)
                print("第{}个batch测试集的accuracy为{}".format(index, accuracy))
