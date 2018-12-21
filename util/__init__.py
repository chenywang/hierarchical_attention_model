# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import re


def get_sentences_from_paragraph(para):
    para = cleaning(para)

    sentences = re.split(r"[,.?!]", str(para))

    # 去除空句子
    sentences = list(filter(lambda x: len(x) != 0, sentences))

    return sentences


def cleaning(text):
    text = text.lower()  # 小写
    text = text.replace('\r\n', " ")  # 新行，我们是不需要的
    text = text.replace('\n', " ")  # 新行，我们是不需要的
    text = text.replace('\r', " ")  # 新行，我们是不需要的
    text = re.sub(r"-", " ", text)  # 把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub('，', ',', text)  #
    text = re.sub('。', '.', text)  #
    text = re.sub('！', '!', text)  #
    text = re.sub('？', '?', text)  #

    return text
