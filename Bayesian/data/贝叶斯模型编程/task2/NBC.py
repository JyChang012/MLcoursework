# coding: utf-8
import os
import time
import random
import jieba
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib import pyplot as plt
import pandas as pd


def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def MakeWordsSet(words_file):
    """
    函数说明:读取文件里的内容，并去重
    Parameters:
        words_file - 文件路径
    Returns:
        words_set - 读取的内容的set集合
    """
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0 and word not in words_set:  # 去重
                words_set.add(word)
    return words_set


def TextProcessing(folder_path, test_size=0.2):
    """
    函数说明:中文文本处理
    Parameters:
        folder_path - 文本存放的路径
        test_size - 测试集占比，默认占所有数据集的百分之20
    Returns:
        all_words_list - 按词频降序排序的训练集列表
        train_data_list - 训练集列表
        test_data_list - 测试集列表
        train_class_list - 训练集标签列表
        test_class_list - 测试集标签列表
    """
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        if folder[0] is not '.':
            new_folder_path = os.path.join(folder_path, folder)
            files = os.listdir(new_folder_path)
            # 类内循环
            j = 1
            for file in files:
                if j > 100:  # 每类text样本数最多100
                    break
                with open(os.path.join(new_folder_path, file), 'r') as fp:
                    raw = fp.read()

                word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
                word_list = list(word_cut)  # genertor转化为list，每个词unicode格式

                data_list.append(word_list)
                class_list.append(folder)
                j += 1

    # 划分训练集和测试集
    data_class_list = list(zip(data_list, class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1  # Length of test size
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # 统计词频放入all_words_dict, 仅统计训练集
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:  # wrong
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序，词频从高到低
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list))[0]

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list, deleteN, stopwords_set=set(), max_feature_num=1000):
    """
    函数说明:文本特征选取
    Parameters:
        all_words_list - 训练集所有文本列表
        deleteN - 删除词频最高的deleteN个词
        stopwords_set - 指定的结束语
    Returns:
        feature_words - 特征集
    """
    # 选取特征词
    feature_words = []
    n = 1

    all_words_list = list(all_words_list)
    for word in all_words_list.copy():
        if not check_contain_chinese(word):
            all_words_list.remove(word)

    for t in range(deleteN, len(all_words_list), 1):
        if n > max_feature_num:  # feature_words的维度1000
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为feature_word
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words):
    """
    函数说明:根据feature_words将文本向量化
    Parameters:
        train_data_list - 训练集
        test_data_list - 测试集
        feature_words - 特征集
    Returns:
        train_feature_list - 训练集向量化列表
        test_feature_list - 测试集向量化列表
    """

    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    """
    函数说明:分类器
    Parameters:
        train_feature_list - 训练集向量化的特征文本
        test_feature_list - 测试集向量化的特征文本
        train_class_list - 训练集分类标签
        test_class_list - 测试集分类标签
    Returns:
        test_accuracy - 分类器精度
    """

    """
    
        请编写这部分代码

    """
    classifier = MultinomialNB()
    classifier.fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)

    return test_accuracy


def score(delete_n=0, test_size=0.2, max_feature_num=1000, tf_idf=False, sublinear_tf=False):
    """
    Run the test once, and return the test accuracy.
    :param delete_n: number of deleted most frequent feature word
    :param test_size: portion of test set size
    :param max_feature_num: max number of feature word
    :return: accuracy
    """
    # 文本预处理
    folder_path = './Database/SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,
                                                                                                        test_size=
                                                                                                        test_size)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    # 文本特征提取和分类
    feature_words = words_dict(all_words_list, delete_n, stopwords_set, max_feature_num=max_feature_num)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)

    # Transform the feature list
    if tf_idf is True:
        transformer = TfidfTransformer(sublinear_tf=sublinear_tf)
        data = transformer.fit_transform(train_feature_list + test_feature_list)
        train_feature_list, test_feature_list = data[:len(train_feature_list)], data[len(train_feature_list):]

    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    return test_accuracy


def trans():
    delete_range = list(range(80, 151, 10))
    feature_range = list(range(800, 1301, 100))
    df = pd.DataFrame([], index=delete_range, columns=feature_range)

    for delete_n in delete_range:
        for feature_n in feature_range:
            df.loc[delete_n, feature_n] = score(delete_n=delete_n, max_feature_num=feature_n,
                                                tf_idf=False, sublinear_tf=True)

    print(df)
    return df


if __name__ == '__main__':
    df = trans()
    pass
