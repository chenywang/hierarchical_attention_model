import argparse
import os
import pickle as pl

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk import RegexpTokenizer

from config import train_review_path, embedding_path, embedding_pickle_path, train_path, test_path


def build_emb_matrix_and_vocab(embedding_model, vocabulary_size=20000, embedding_size=100):
    # 0 th element is the default vector for unknowns.
    emb_matrix = np.zeros((vocabulary_size + 2, embedding_size))
    word2index = {}
    index2word = {}
    for k in range(1, vocabulary_size + 1):
        word = embedding_model.wv.index2word[k - 1]
        emb_matrix[k] = embedding_model[word]
        word2index[word] = k
        index2word[k] = word
    word2index['UNK'] = 0
    index2word[0] = 'UNK'
    word2index['STOP'] = vocabulary_size + 1
    index2word[vocabulary_size + 1] = 'STOP'
    return emb_matrix, word2index, index2word


def sentence2index(sent, word2index):
    words = sent.strip().split(' ')
    sent_index = [word2index.get(word, 0) for word in words]
    return sent_index


def get_sentence(index2word, sen_index):
    return ' '.join([index2word[index] for index in sen_index])


def get_train_data(paragraph, word2index):
    sentences = paragraph.split('.')
    sentences = list(filter(lambda s: len(s) != 0, sentences))
    if sentences:
        return [sentence2index(sentence, word2index) for sentence in sentences]
    else:
        return [[]]


def positive_and_negative(line):
    if line['desc_scr'] + line['lgst_scr'] + line['serv_scr'] <= 13 and \
            (line['desc_scr'] < 3 or line['lgst_scr'] < 3 or line['serv_scr'] < 3):
        return 1
    else:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some important parameters.')
    parser.add_argument('-s', '--max_sentence_length', type=int, default=70,
                        help='fix the sentence length in all reviews')
    parser.add_argument('-r', '--max_rev_length', type=int, default=15,
                        help='fix the maximum review length')

    # 获得数据
    data = pd.read_csv(train_review_path, sep='\t')

    # token化数据
    tokenizer = RegexpTokenizer(r'\w+')
    data['tokenize'] = data['context'].apply(tokenizer.tokenize)
    train = list(data['tokenize'])

    # 训练embedding层
    embedding_size = 100
    if os.path.isfile(embedding_path):
        embedding_model = Word2Vec.load(embedding_path)
    else:
        print("保存gensim的word2vec模型")
        embedding_model = Word2Vec(train, size=embedding_size, window=5, min_count=5)
        embedding_model.save(embedding_path)

    # 测试直观效果
    word1 = "好"
    word2 = "坏"
    print("{}的相似词汇是:".format(word1))
    print(embedding_model.most_similar(word1))
    print("{}的相似词汇是:".format(word2))
    print(embedding_model.most_similar(word2))

    args = parser.parse_args()
    max_sentence_length = args.max_sentence_length
    max_rev_length = args.max_rev_length

    print("保存embedding层...")
    emb_matrix, word2index, index2word = build_emb_matrix_and_vocab(embedding_model)
    pl.dump([emb_matrix, word2index, index2word], open(embedding_pickle_path, "wb"))

    # 获得数据
    data['label'] = data.apply(positive_and_negative, axis=1)
    data = data[['context', 'label']].sample(frac=1.0).reset_index(drop=True)

    # 分为训练与测试样本
    split_fraction = 0.75
    split_index = int(0.75 * data.shape[0])
    train_data, test_data = data[:split_index], data[split_index:]

    print('保存训练样本与测试样本')
    train_data.to_csv(train_path, sep='\t', header=True, index=False)
    test_data.to_csv(test_path, sep='\t', header=True, index=False)
