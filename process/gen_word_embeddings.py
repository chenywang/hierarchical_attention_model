import os

import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer

from config import config

if __name__ == "__main__":

    # 获得数据
    position_data = pd.read_csv(config.positive_review_path, sep='\t')
    negative_data = pd.read_csv(config.negative_review_path, sep='\t')
    data = position_data.append(negative_data)

    # token化数据
    tokenizer = RegexpTokenizer(r'\w+')
    data['context'] = data['context'].apply(tokenizer.tokenize)
    train = list(data['context'])

    # 训练embedding层
    embedding_size = 100
    if os.path.isfile(config.embedding_path):
        embedding_model = Word2Vec.load(config.embedding_path)
    else:
        embedding_model = Word2Vec(train, size=embedding_size, window=5, min_count=5)
        embedding_model.save(config.embedding_path)

    # 测试直观效果
    word1 = "好"
    word2 = "坏"
    print("{}的相似词汇是:".format(word1))
    print(embedding_model.most_similar(word1))
    print("{}的相似词汇是:".format(word2))
    print(embedding_model.most_similar(word2))
