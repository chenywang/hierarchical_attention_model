# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import math

import numpy as np
import pandas as pd
from tensorflow.contrib.keras import preprocessing

from preprocess import get_train_data


def review_to_tensor(review_list, word2index, max_sentence_length, max_review_length):
    """
    As the result, each review will be composed of max_rev_len sentences. If the original review is longer than that,
    we truncate it, and if shorter than that, we append empty sentences to it. And each sentence will be composed of
    sent_length words. If the original sentence is longer than that, we truncate it, and if shorter, we append the word
    of 'UNK' to it. Also, we keep track of the actual number of sentences each review contains.
    :param review_list:
    :param word2index
    :param max_sentence_length:
    :param max_review_length:
    :return: [batch_size, max_review_length, max_sentence_length]
    """
    batch_size = len(review_list)
    review_tensor_list = np.zeros((batch_size, max_review_length, max_sentence_length), dtype=np.int32)
    review_lens = []
    for index, review in enumerate(review_list):
        review_tensor = get_train_data(review, word2index)
        review_tensor = preprocessing.sequence.pad_sequences(review_tensor, maxlen=max_sentence_length,
                                                             padding="post", truncating="post",
                                                             value=0)
        review_lens.append(min(review_tensor.shape[0], 15))
        review_tensor = preprocessing.sequence.pad_sequences([review_tensor], maxlen=max_review_length,
                                                             padding="post", truncating="post",
                                                             value=np.zeros(max_sentence_length))[0]
        review_tensor_list[index] = review_tensor

    return review_tensor_list, np.array(review_lens)


def gen_batch_train_data(data, word2index, max_sentence_length, max_review_length, data_path=None, batch_size=512,
                         shuffle=True):
    if data is None:
        data = pd.read_csv(data_path, sep='\t')
    times = int(math.ceil(data.shape[0] / batch_size))
    for i in range(times):
        batch_data = data[i * batch_size: (i + 1) * batch_size]
        if shuffle:
            batch_data = batch_data.sample(frac=1).reset_index(drop=True)

        batch_x_data = np.array(batch_data['context'])
        batch_x_data, batch_data_review_lens = review_to_tensor(batch_x_data, word2index, max_sentence_length,
                                                                max_review_length)
        batch_y_data = np.array(batch_data['label'])
        yield batch_x_data, batch_y_data, batch_data_review_lens
