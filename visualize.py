# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import argparse
import pickle as pl

import pandas as pd
import tensorflow as tf

from config import config
from algorithm.implement.components import visualize
from util.data_utils import gen_batch_train_data
from attention_model.hierarchical_attention_model import HierarchicalModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for building the model.')
    parser.add_argument('-m', '--max_sentence_length', type=int, default=70,
                        help='fix the sentence length in all reviews, 需要跟预处理保持一致')
    parser.add_argument('-M', '--max_rev_length', type=int, default=15,
                        help='fix the maximum review length, 需要跟预处理保持一致')

    # 设置基本参数
    args = parser.parse_args()
    max_sentence_length = args.max_sentence_length
    max_review_length = args.max_rev_length
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
    test_data = pd.read_csv(config.test_path, sep='\t')
    test_data['review_length'] = test_data['context'].apply(lambda review: len(review.split('.')))
    test_data = test_data[test_data['review_length'] > 4][:test_size]
    review_list = list(test_data['context'])
    y_list = list(test_data['label'])
    batch_data, batch_label, review_length_list = next(gen_batch_train_data(test_data,
                                                                            word2index,
                                                                            max_sentence_length,
                                                                            max_review_length,
                                                                            batch_size=test_size,
                                                                            shuffle=False))

    with tf.Session() as sess:
        print("载入模型中...")
        latest_cpt_file = tf.train.latest_checkpoint(config.log_path)
        model.restore(sess, saver, latest_cpt_file)
        review_probability_list, alphas_words_matrix, alphas_sentences_matrix = model.predict(sess, batch_data,
                                                                                              review_length_list)
    alphas_words_matrix = alphas_words_matrix.reshape((test_size, max_review_length, max_sentence_length))
    print('生成可视化h5中...')
    for index, review, alphas_words_list, alphas_sentences_list, y, probability in \
            zip(range(test_size), review_list, alphas_words_matrix, alphas_sentences_matrix, y_list,
                review_probability_list):
        visualize(review, alphas_words_list, alphas_sentences_list, index, y, probability)
    print('生成完毕')