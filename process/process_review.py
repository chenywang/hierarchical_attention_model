# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import pickle as pl

import pandas as pd
import tensorflow as tf

from algorithm.implement.components import visualize
from attention_model.hierarchical_attention_model import HierarchicalModel
from config import config
from segment.impl.segmentor import MySegmentor
from util.data_utils import gen_batch_train_data


class Processor:
    def __init__(self):
        print("载入嵌入层...")
        self.emb_matrix, self.word2index, self.index2word = pl.load(open(config.embedding_pickle_path, "rb"))

        print("生成模型中...")
        self.max_sentence_length = 70
        self.max_review_length = 15
        self.model = HierarchicalModel(self.max_sentence_length, self.max_review_length, self.emb_matrix)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        latest_cpt_file = tf.train.latest_checkpoint(config.log_path)
        self.model.restore(self.sess, self.saver, latest_cpt_file)

        self.segmentor = MySegmentor()

    def process_review(self, review):
        # review = '面膜质量很好，用过感觉皮肤水水的很嫩，很温和不刺激皮肤，还起到美的的效果，下次还来这家店！'

        sentences = self.segmentor.segment_paragraph(review)
        review = [' '.join(sentence) for sentence in sentences]
        review = '.'.join(review)
        data = pd.DataFrame([{'context': review, 'label': 0}])
        data, _, review_length = next(gen_batch_train_data(data,
                                                           self.word2index,
                                                           self.max_sentence_length,
                                                           self.max_review_length,
                                                           batch_size=1,
                                                           shuffle=False))
        # 模型预测
        probability, alphas_words_list, alphas_sentences_list = self.model.predict(self.sess, data, review_length)

        # 输出h5
        alphas_words_list = alphas_words_list.reshape((1, self.max_review_length, self.max_sentence_length))
        return visualize(review, alphas_words_list[0], alphas_sentences_list[0], 0, 0, probability[0],
                         write_actual=False,
                         write_h5=False)
