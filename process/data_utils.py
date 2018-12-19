# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import math

import numpy as np
import pandas as pd

from process.preprocess_reviews import preprocess_review, get_train_data


def gen_batch_train_data(data, word2index, sentence_length, max_rev_length, data_path=None, batch_size=512):
    if data is None:
        data = pd.read_csv(data_path, sep='\t')
    times = int(math.ceil(data.shape[0] / batch_size))
    for i in range(times):
        batch_data = data[i * batch_size: (i + 1) * batch_size].sample(frac=1).reset_index(drop=True)
        batch_x_data = list(batch_data['context'].apply(get_train_data, word2index=word2index))
        batch_x_data, batch_data_review_lens = preprocess_review(batch_x_data, sentence_length, max_rev_length)
        batch_y_data = list(batch_data['label'])
        # yield pd.DataFrame({'review': batch_x_data, 'label': batch_y_data, 'length': batch_data_review_lens})
        yield np.asarray(batch_x_data), batch_y_data, batch_data_review_lens
