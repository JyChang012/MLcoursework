import pandas as pd
import numpy as np


class NaiveBayes:

    def __init__(self, data=pd.DataFrame([])):
        self.data = data
        self.label_counts = data.value_counts()

    def discriminate(self, test_set=pd.DataFrame()):
        labels = []
        for idx in test_set.index:
            label = self._discriminate(test_set.loc[idx])
            labels.append(label)
        return pd.Series(labels, index=test_set.index)

    def _discriminate(self, test_sample=pd.Series()):
        label2prob = {}
        for label in self.data.iloc[:, -1].unique():
            p_c = self.label_counts[label] / self.data.shape[0]
            p_c2x = 1
            for attr in test_sample.index:
                p_label2xi = self.data.loc[self.data[attr] == test_sample[attr] & self.data.iloc[:, -1] == label].shape[
                                 0] / self.label_counts[label]
                p_c2x *= p_label2xi

            h_nb = p_c * p_c2x
            label2prob[label] = h_nb

        optml_label = max(label2prob, key=label2prob.get)
        return optml_label

