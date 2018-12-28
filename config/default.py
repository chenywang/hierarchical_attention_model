# -*- coding: utf-8 -*-
# @Author: disheng
import os

PROJECT_PATH = os.path.abspath(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.pardir))


project_path = PROJECT_PATH
data_path = PROJECT_PATH + '/data/'
review_path = data_path + 'reviews/'
embedding_path = review_path + 'review_embedding.model'
embedding_pickle_path = review_path + 'review_embedding_layer.pickle'
positive_review_path = os.path.join(review_path, 'positive_review.csv')
train_review_path = os.path.join(review_path, 'review.csv')
negative_review_path = os.path.join(review_path, 'negative_review.csv')
train_path = os.path.join(review_path, "train_data.csv")
test_path = os.path.join(review_path, "test_data.csv")

imbd_path = data_path + 'aclImdb'

log_path = project_path + '/logs/'
visualization_path = project_path + '/visualization/'

ltp_model_path = '/Users/michael-wang/pdd_projects/data/ltp_data/'
seg_model_path = ltp_model_path + 'cws.model'
user_dict_path = project_path + '/customize_dict/user_dict.dict'


