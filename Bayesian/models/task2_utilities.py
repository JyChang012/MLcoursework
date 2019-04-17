import pandas as pd
import numpy as np
import os
import utils_task1 as utl
from sklearn.utils import shuffle
import naive_bayes as nb
from matplotlib import pyplot as plt
from Bayesian.data.贝叶斯模型编程.task2.NBC import score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import TfidfTransformer
import jieba as jb
import os


def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def text2list(text=''):
    text_cut = jb.lcut(text, cut_all=False)
    for word in text_cut.copy():
        if not check_contain_chinese(word):
            text_cut.remove(word)
    return text_cut




def task2_preprocess():
    folders_path = './Database/SogouC/Sample'
    class2strs = {}

    for folder in os.listdir('./Database/SogouC/Sample'):
        if folder[0] is not '.':
            class2strs[folder] = []
            for file in os.listdir(os.path.join(folders_path, folder)):
                file_path = os.path.join(folders_path, folder, file)
                with open(file_path, encoding='utf-8') as f:
                    text = f.read()
                text_cut = text2list(text)
                class2strs[folder].append(text_cut)

    all_word_list = set()
    for c in class2strs:
        for word_list in class2strs[c]:
            all_word_list = all_word_list | set(word_list)

    pass

if __name__ == '__main__':
    task2_preprocess()