# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import math

import numpy as np
import pandas as pd
from tensorflow.contrib.keras import preprocessing

from process.preprocess import get_train_data


def review_to_tensor(review_list, word2index, max_sentence_length, max_review_length, keep_in_dict=20000):
    """
    As the result, each review will be composed of max_rev_len sentences. If the original review is longer than that,
    we truncate it, and if shorter than that, we append empty sentences to it. And each sentence will be composed of
    sent_length words. If the original sentence is longer than that, we truncate it, and if shorter, we append the word
    of 'UNK' to it. Also, we keep track of the actual number of sentences each review contains.
    :param review_list:
    :param word2index
    :param max_sentence_length:
    :param max_review_length:
    :param keep_in_dict:
    :return: [batch_size, max_review_length, max_sentence_length]
    """

    review_tensor_list = []
    review_lens = []
    for review in review_list:
        review_tensor = get_train_data(review, word2index)
        review_tensor = preprocessing.sequence.pad_sequences(review_tensor, maxlen=max_sentence_length,
                                                             padding="post", truncating="post",
                                                             value=0)
        review_lens.append(review_tensor.shape[0])
        append_sentence_count = 0 if review_tensor.shape[0] >= max_review_length else \
            max_review_length - review_tensor.shape[0]
        append_sentence = np.zeros((append_sentence_count, max_sentence_length), dtype=np.int32)
        review_tensor = np.append(review_tensor, append_sentence, axis=0)
        review_tensor_list.append(review_tensor)

    return review_tensor_list, review_lens


def gen_batch_train_data(data, word2index, sentence_length, max_rev_length, data_path=None, batch_size=512, shuffle=True):
    if data is None:
        data = pd.read_csv(data_path, sep='\t')
    times = int(math.ceil(data.shape[0] / batch_size))
    for i in range(times):
        batch_data = data[i * batch_size: (i + 1) * batch_size]
        if shuffle:
            batch_data = batch_data.sample(frac=1).reset_index(drop=True)

        batch_x_data = list(batch_data['context'])
        batch_x_data, batch_data_review_lens = review_to_tensor(batch_x_data, word2index, sentence_length,
                                                                max_rev_length,
                                                                len(word2index))
        batch_y_data = list(batch_data['label'])
        yield np.asarray(batch_x_data), batch_y_data, batch_data_review_lens
