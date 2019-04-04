import re


def text_parse(big_string=''):  # input is big string, #output is word list
    """
    接受一个大字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串转换为小写。
    """
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def create_vocab_list(data_set):
    """
    创建一个包含在所有文档中出现的不重复的词的列表。
    """
    vocab_set = set([])  # create empty set
    for document in data_set:
        vocab_set = vocab_set | set(document)  # union of the two sets
    return list(vocab_set)


def bag_of_words2Vec(vocab_list, input_set):
    """
    获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数.
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec
