# -*- coding:utf-8 -*-
# @Author : Michael-Wang

def redistribute(values):
    s = sum(values)
    return [float(v) / s for v in values]
