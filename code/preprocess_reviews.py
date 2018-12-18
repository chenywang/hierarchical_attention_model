import argparse
import os
import pickle as pl
import random
import re
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
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


def gen_data(data_dir, word2index):
    data = []
    tokenizer = RegexpTokenizer(r'\w+|[?!]|\.{1,}')
    for filename in os.listdir(data_dir):
        file = os.path.join(data_dir, filename)
        with open(file) as f:
            content = f.readline().lower()
            content_formatted = ' '.join(tokenizer.tokenize(content))[:-1]
            sents = re.compile("[?!]|\.{1,}").split(content_formatted)
            sents_index = [sentence2index(sent, word2index) for sent in sents]
            data.append(sents_index)
    return data


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
    combination_list = [(line[0], line[1], line[2])for line in zip(x_data, y_data, data_review_lens)]
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

    # print('sent length is set as {}'.format(sentence_length))
    # print('rev length is set as {}'.format(max_rev_length))
    # working_dir = config.imbd_path
    # train_dir = os.path.join(working_dir, "train")
    # train_pos_dir = os.path.join(train_dir, "pos")
    # train_neg_dir = os.path.join(train_dir, "neg")
    # test_dir = os.path.join(working_dir, "test")
    # test_pos_dir = os.path.join(test_dir, "pos")
    # test_neg_dir = os.path.join(test_dir, "neg")
    #
    # if os.path.isfile(config.embedding_path):
    #     embedding_model = Word2Vec.load(config.embedding_path)
    # else:
    #     raise ValueError("please run gen_word_embeddings.py first to generate embeddings!")
    # print("generate word to index dictionary and inverse dictionary...")
    # emb_matrix, word2index, index2word = build_emb_matrix_and_vocab(embedding_model)
    # print("format each review into sentences, and also represent each word by index...")
    # train_pos_data = gen_data(train_pos_dir, word2index)
    # train_neg_data = gen_data(train_neg_dir, word2index)
    # train_data = train_neg_data + train_pos_data
    # test_pos_data = gen_data(test_pos_dir, word2index)
    # test_neg_data = gen_data(test_neg_dir, word2index)
    # test_data = test_neg_data + test_pos_data
    #
    # print("preprocess each review...")
    # x_train, train_review_lens = preprocess_review(train_data, sentence_length, max_rev_length)
    # x_test, test_review_lens = preprocess_review(test_data, sentence_length, max_rev_length)
    # y_train = [0] * len(train_neg_data) + [1] * len(train_pos_data)
    # y_test = [0] * len(test_neg_data) + [1] * len(test_pos_data)
    #
    # print("save word embedding matrix ...")
    # emb_filename = os.path.join(working_dir, "emb_matrix")
    # # emb_matrix.dump(emb_filename)
    # pl.dump([emb_matrix, word2index, index2word], open(emb_filename, "wb"))
    #
    # print("save review data for training...")
    # df_train = pd.DataFrame({'review': x_train, 'label': y_train, 'length': train_review_lens})
    # train_filename = os.path.join(working_dir, "train_df_file")
    # df_train.to_pickle(train_filename)
    #
    # print("save review data for testing...")
    # df_test = pd.DataFrame({'review': x_test, 'label': y_test, 'length': test_review_lens})
    # test_filename = os.path.join(working_dir, "test_df_file")
    # df_test.to_pickle(test_filename)

    ########

    # 载入embedding层
    if os.path.isfile(config.embedding_path):
        embedding_model = Word2Vec.load(config.embedding_path)
    else:
        raise ValueError("please run gen_word_embeddings.py first to generate embeddings!")
    print("generate word to index dictionary and inverse dictionary...")
    emb_matrix, word2index, index2word = build_emb_matrix_and_vocab(embedding_model)

    # 获得数据
    size = 100000
    position_data = pd.read_csv(config.positive_review_path, sep='\t')
    negative_data = pd.read_csv(config.negative_review_path, sep='\t')
    position_data = position_data[:size]
    negative_data = negative_data[:size]
    data = position_data.append(negative_data)
    x_data = list(data['context'].apply(get_train_data, word2index=word2index))
    y_data = [0] * position_data.shape[0] + [1] * negative_data.shape[0]

    # 预处理
    x_data, data_review_lens = preprocess_review(x_data, sentence_length, max_rev_length)

    # 分为训练与测试样本
    x_train, y_train, train_review_lens, x_test, y_test, test_review_lens = \
        split_train_and_test(x_data, y_data, data_review_lens)

    # 保存embedding层
    pl.dump([emb_matrix, word2index, index2word], open(config.embedding_pickle_path, "wb"))

    print("save review data for training...")
    df_train = pd.DataFrame({'review': x_train, 'label': y_train, 'length': train_review_lens})
    df_train['review'] = df_train['review'].apply(lambda x: json.dumps(x.tolist()))
    df_train.to_csv(config.train_path, index=False, sep='\t', header=True)
    # df_train.to_pickle(config.train_path, protocol=4)

    print("save review data for testing...")
    df_test = pd.DataFrame({'review': x_test, 'label': y_test, 'length': test_review_lens})
    df_test['review'] = df_test['review'].apply(lambda x: json.dumps(x.tolist()))
    df_train.to_csv(config.test_path, index=False, sep='\t', header=True)
    # df_test.to_pickle(config.test_path, protocol=4)
