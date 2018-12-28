#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:Michael-Wang

from pyltp import Segmentor

from config import seg_model_path, user_dict_path
from util import get_sentences_from_paragraph


class MySegmentor:

    def __init__(self):
        self.segmentor = Segmentor()
        self.load_model()

    def load_config(self, config):
        """载入配置"""
        pass

    def load_model(self):
        """载入ltp分词模型"""
        # 修改模型所在位置
        self.seg_model_path = seg_model_path

        # 修改字典地址
        self.user_dict_path = user_dict_path

        # 载入模型
        self.segmentor.load_with_lexicon(
            self.seg_model_path, self.user_dict_path)

    def release_model(self):
        """释放ltp分词模型"""
        self.segmentor.release()

    def segment_paragraph(self, paragraph):
        """
        将段落进行按句子进行分词
        :param paragraph: 某个段落
        :return: [[word1, word2], [words1, words2]]
        """
        # 获得句子列表
        sentences = get_sentences_from_paragraph(paragraph)
        result = list()

        # 遍历每条句子，进行标签提取
        for sentence in sentences:
            sub_result = self.segment_sentence(sentence)
            result.append(sub_result)
        return result

    def segment_sentence(self, sentence):
        seg_list = self.segmentor.segment(sentence)
        result = list()
        for seg in seg_list:
            result.append(seg)
        return result


if __name__ == "__main__":
    segmentor = MySegmentor()
    sentences = segmentor.segment_paragraph("面膜质量很好，用过感觉皮肤水水的很嫩，很温和不刺激皮肤，还起到美的的效果，下次还来这家店！")
    result = [' '.join(sentence) for sentence in sentences]
    result = '.'.join(result)
    print(result)
