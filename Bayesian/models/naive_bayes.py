import pandas as pd
import numpy as np
from scipy.stats import norm

sum = np.sum


class NaiveBayes:

    def __init__(self, laplacian_correction=False, continuous_col=pd.Series([])):
        self.y = None
        self.X = None
        self.laplacian_correction = laplacian_correction
        self.continuous_col = continuous_col

    def fit(self, X=pd.DataFrame(), y=pd.Series([])):
        self.X = X
        self.y = y

    def discriminate(self, X=pd.DataFrame()):
        y = []
        for idx in X.index:
            label = self._discriminate(X.loc[idx])
            y.append(label)
        return pd.Series(y, index=X.index)

    def _discriminate(self, x=pd.Series()):
        label2prob = {}
        if self.laplacian_correction is False:
            for y_val in self.y.unique():
                p_y = self.y.value_counts(sort=False).loc[y_val] / self.y.shape[0]
                p_y2x = 1
                for xi in x.index:
                    if xi not in self.continuous_col:
                        p_y2xi = sum((self.y == y_val) & (self.X[xi] == x.loc[xi])) / sum(self.y == y_val)
                        p_y2x *= p_y2xi
                    else:
                        mean = self.X.loc[(self.y == y_val), xi].mean()
                        var = self.X.loc[(self.y == y_val), xi].var()

                        if var == 0 and mean == 0:
                            p_y2xi = sum((self.y == y_val) & (self.X[xi] == x.loc[xi])) / sum(self.y == y_val)
                        else:
                            p_y2xi = norm.pdf(x.loc[xi], mean, np.sqrt(var))
                        p_y2x *= p_y2xi

                h_nb = p_y * p_y2x
                label2prob[y_val] = h_nb

        else:
            for y_val in self.y.unique():
                p_y = (self.y.value_counts(sort=False).loc[y_val] + 1) / (self.y.shape[0] + self.y.unique().shape[0])
                p_y2x = 1
                for xi in x.index:
                    if xi not in self.continuous_col:
                        p_y2xi = (sum((self.y == y_val) & (self.X[xi] == x.loc[xi])) + 1) /\
                                 (sum(self.y == y_val) + self.X[xi].unique().shape[0])
                        p_y2x *= p_y2xi
                    else:
                        mean = self.X.loc[(self.y == y_val), xi].mean()
                        var = self.X.loc[(self.y == y_val), xi].var()

                        if mean == 0 and var == 0:
                            p_y2xi = (sum((self.y == y_val) & (self.X[xi] == x.loc[xi])) + 1) /\
                                 (sum(self.y == y_val) + self.X[xi].unique().shape[0])
                        else:
                            p_y2xi = norm.pdf(x.loc[xi], mean, np.sqrt(var))

                        p_y2x *= p_y2xi

                h_nb = p_y * p_y2x
                label2prob[y_val] = h_nb

        optml_label = max(label2prob, key=label2prob.get)
        return optml_label

    def score(self, X=pd.DataFrame(), y=pd.Series()):
        y_estimator = self.discriminate(X)
        return sum(y_estimator == y) / y.shape[0]
