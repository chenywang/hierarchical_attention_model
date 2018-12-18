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

    imbd_path = data_path + 'aclImdb'

