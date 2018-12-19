import argparse
import json
import os
import pickle as pl
import random

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.contrib.keras import preprocessing

from config import config


def build_emb_matrix_and_vocab(embedding_model, keep_in_dict=20000, embedding_size=100):
    # 0 th element is the default vector for unknowns.
    emb_matrix = np.zeros((keep_in_dict + 2, embedding_size))
    word2index = {}
    index2word = {}
    for k in range(1, keep_in_dict + 1):
        word = embedding_model.wv.index2word[k - 1]
        emb_matrix[k] = embedding_model[word]
        word2index[word] = k
        index2word[k] = word
    word2index['UNK'] = 0
    index2word[0] = 'UNK'
    word2index['STOP'] = keep_in_dict + 1
    index2word[keep_in_dict + 1] = 'STOP'
    return emb_matrix, word2index, index2word


def sentence2index(sent, word2index):
    words = sent.strip().split(' ')
    sent_index = [word2index[word] if word in word2index else 0 for word in words]
    return sent_index


def get_sentence(index2word, sen_index):
    return ' '.join([index2word[index] for index in sen_index])


def get_train_data(paragraph, word2index):
    sentences = paragraph.split('.')
    sentences = filter(lambda s: len(s) != 0, sentences)
    return [sentence2index(sentence, word2index) for sentence in sentences]


def preprocess_review(data, sent_length, max_rev_len, keep_in_dict=10000):
    ## As the result, each review will be composed of max_rev_len sentences. If the original review is longer than that, we truncate it, and if shorter than that, we append empty sentences to it. And each sentence will be composed of sent_length words. If the original sentence is longer than that, we truncate it, and if shorter, we append the word of 'UNK' to it. Also, we keep track of the actual number of sentences each review contains.
    data_formatted = []
    review_lens = []
    for review in data:
        review_formatted = preprocessing.sequence.pad_sequences(review, maxlen=sent_length, padding="post",
                                                                truncating="post", value=keep_in_dict + 1)
        review_len = review_formatted.shape[0]
        review_lens.append(review_len if review_len <= max_rev_len else max_rev_len)
        lack_len = max_rev_length - review_len
        review_formatted_right_len = review_formatted
        if lack_len > 0:
            # extra_rows = np.zeros([lack_len, sent_length], dtype=np.int32)
            extra_rows = np.full((lack_len, sent_length), keep_in_dict + 1)
            review_formatted_right_len = np.append(review_formatted, extra_rows, axis=0)
        elif lack_len < 0:
            row_index = [max_rev_length + i for i in list(range(0, -lack_len))]
            review_formatted_right_len = np.delete(review_formatted, row_index, axis=0)
        data_formatted.append(review_formatted_right_len)
    return data_formatted, review_lens


def split_train_and_test(x_data, y_data, data_review_lens, split_fraction=0.75):
    combination_list = [(line[0], line[1], line[2]) for line in zip(x_data, y_data, data_review_lens)]
    random.shuffle(combination_list)
    size = len(combination_list)
    train_index = int(size * split_fraction)
    x_train_data = [e[0] for e in combination_list[:train_index]]
    y_train_data = [e[1] for e in combination_list[:train_index]]
    train_review_lens = [e[2] for e in combination_list[:train_index]]

    x_test_data = [e[0] for e in combination_list[train_index:]]
    y_test_data = [e[1] for e in combination_list[train_index:]]
    test_review_lens = [e[2] for e in combination_list[train_index:]]
    return x_train_data, y_train_data, train_review_lens, x_test_data, y_test_data, test_review_lens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some important parameters.')
    parser.add_argument('-s', '--sentence_length', type=int, default=70,
                        help='fix the sentence length in all reviews')
    parser.add_argument('-r', '--max_rev_length', type=int, default=15,
                        help='fix the maximum review length')

    args = parser.parse_args()
    sentence_length = args.sentence_length
    max_rev_length = args.max_rev_length

    # 载入并保存embedding层
    if os.path.isfile(config.embedding_path):
        embedding_model = Word2Vec.load(config.embedding_path)
    else:
        raise ValueError("please run gen_word_embeddings.py first to generate embeddings!")
    print("generate word to index dictionary and inverse dictionary...")
    emb_matrix, word2index, index2word = build_emb_matrix_and_vocab(embedding_model)
    pl.dump([emb_matrix, word2index, index2word], open(config.embedding_pickle_path, "wb"))

    # 获得数据
    position_data = pd.read_csv(config.positive_review_path, sep='\t')
    negative_data = pd.read_csv(config.negative_review_path, sep='\t')
    data = position_data.append(negative_data)
    y_data = [0] * position_data.shape[0] + [1] * negative_data.shape[0]
    data['label'] = y_data
    data = data[['context', 'label']].sample(frac=1.0).reset_index(drop=True)

    # 分为训练与测试样本
    split_fraction = 0.75
    split_index = int(0.75 * data.shape[0])
    train_data, test_data = data[:split_index], data[split_index:]

    train_data.to_csv(config.train_path, sep='\t', header=True, index=False)
    test_data.to_csv(config.test_path, sep='\t', header=True, index=False)



