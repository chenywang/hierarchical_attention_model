# -*- coding:utf-8 -*-
# @Author : Michael-Wang
from process.process_review import Processor

if __name__ == '__main__':
    processor = Processor()
    review = '面膜质量很好，用过感觉皮肤水水的很嫩，很温和不刺激皮肤，还起到美的的效果，下次还来这家店！'
    print(processor.process_review(review))
    print(processor.process_review(review))
    print(processor.process_review(review))
