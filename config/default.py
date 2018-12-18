# -*- coding: utf-8 -*-
# @Author: disheng
import os

PROJECT_PATH = os.path.abspath(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.pardir))


class Config(object):
    # 项目路径
    project_path = PROJECT_PATH
    data_path = PROJECT_PATH + '/data/'
    review_path = data_path + 'reviews/'
    embedding_path = review_path + 'review_embedding.model'
    embedding_pickle_path = review_path + 'review_embedding.pickle'
    positive_review_path = os.path.join(review_path, 'positive_review.csv')
    negative_review_path = os.path.join(review_path, 'negative_review.csv')
    train_path = os.path.join(review_path, "train_vector.csv")
    test_path = os.path.join(review_path, "text_vector.csv")


    imbd_path = data_path + 'aclImdb'


    log_path = project_path + '/logs/'
    visualization_path = project_path + '/visualization/'





