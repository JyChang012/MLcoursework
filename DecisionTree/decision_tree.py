import numpy as np
import pandas as pd
import pydot as dot
from scipy.stats import entropy
from sklearn.utils import shuffle


class TreeNode:
    """Node of decision tree class."""

    def __init__(self, parent):
        self.attr = None
        self.parent = parent
        self.weights = dict()
        self.children = dict()
        self.label = None  # If label is None, that node is not a leave
        self.is_continuous = False  # False or store the optimal partition point.
        self.data = parent.data

    def set_label(self, label):
        self.label = label

    def discriminate(self, test_point=pd.Series([])):
        if not bool(self.children):
            return [self.label, 1]
        else:
            try:
                if self.attr in test_point.index and test_point.loc[self.attr] is not np.nan:
                    if not self.is_continuous:
                        return self.children[test_point.loc[self.attr]].discriminate(test_point)
                    else:
                        return self.children[test_point.loc[self.attr] > self.is_continuous].discriminate(test_point)
                else:
                    weights_dict = dict()
                    for child in self.children:
                        return_val = self.children[child].discriminate(test_point)
                        weights_dict[child] = [return_val[0], return_val[1] * self.weights[child]]
                    optml_child = max(weights_dict, key=lambda x: weights_dict.get(x)[1])
                    return weights_dict[optml_child]
            except KeyError:
                return 'no_match'

    def add_child(self, value, child):
        self.children[value] = child


class DecisionTree:
    """Decision tree class."""

    def __init__(self, data=pd.DataFrame([]), type='ID3', continuous_col=pd.Series([]), prune_rate=None,
                 prune_type='post'):
        if not data.empty:
            self.type = type
            self.weight = dict()
            self.data = data
            self.continuous_col = continuous_col
            if prune_rate is None:
                self.root = tree_generate_recursion(data, type, continuous_col, parent=self)
            else:
                train_size = int(data.shape[0] * (1 - prune_rate))
                shuffle_data = shuffle(data)
                if prune_type is 'post':
                    self.root = tree_generate_recursion(shuffle_data.iloc[:train_size], type,
                                                        continuous_col, parent=self)
                    self.post_pruning(shuffle_data.iloc[train_size:])
                elif prune_type is 'pre':
                    self.root = tree_generate_recursion(shuffle_data.iloc[:train_size], type,
                                                        continuous_col, parent=self)
                    self.pre_pruning(shuffle_data.iloc[train_size:])

    def discriminate(self, test_set=pd.DataFrame([])):
        labels = []
        for idx in test_set.index:
            test_point = test_set.loc[idx]
            labels.append(self.root.discriminate(test_point)[0])
        return pd.Series(labels, name='labels', index=test_set.index)

    def pre_pruning(self, data):
        pre_pruning_recursion(self, self.root, data)

    def post_pruning(self, data):
        post_pruning_recursion(self, self.root, data)

    def visualize(self, file_name='Decision Tree', graph_name='Decision Tree'):
        global name
        graph = dot.Dot(graph_type='digraph', graph_name=graph_name)
        visualize_recursion(self.root, graph)
        graph.write_jpg(file_name + '.jpg')
        name = 0


def pre_pruning_recursion(tree, node, data):
    if not bool(node.children):
        return

    precision_without_prun = (tree.discriminate(data.iloc[:, :-1]) == data.iloc[:, -1]).sum() / data.shape[0]
    temp_children = node.children
    node.children = dict()
    precision_with_prun = (tree.discriminate(data.iloc[:, :-1]) == data.iloc[:, -1]).sum() / data.shape[0]
    if precision_without_prun - precision_with_prun < 0.03:
        node.attr = None
        return
    else:
        node.children = temp_children
        for child in node.children:
            if bool(node.children[child].children):
                pre_pruning_recursion(tree, node.children[child], data)


name = 0  # Global variable used for pydot node names.


def visualize_recursion(node, graph):
    global name
    if not bool(node.children):
        g_node_me = dot.Node(name=name, label=f'leave_node,\nlabel: {node.label}')
        name = name + 1
        graph.add_node(g_node_me)
    else:
        if node.is_continuous is False:  # Discrete case
            g_node_me = dot.Node(name=name, label=f'Attribute: "{node.attr}"')
            name = name + 1
            graph.add_node(g_node_me)
            for child in node.children:
                g_child_node = visualize_recursion(node.children[child], graph)
                edge = dot.Edge(g_node_me, g_child_node, label=f'Attribute "{node.attr}" = {child}')
                graph.add_edge(edge)
        else:
            g_node_me = dot.Node(name=name,
                                 label=f'Continuous attribute: "{node.attr}"\npartition point: {node.is_continuous}')
            name = name + 1
            graph.add_node(g_node_me)
            for child in node.children:
                g_child_node = visualize_recursion(node.children[child], graph)
                edge = dot.Edge(g_node_me, g_child_node, label=f'{node.attr} > {node.is_continuous} is {child}')
                graph.add_edge(edge)
    return g_node_me


def post_pruning_recursion(tree, node, data):
    if not bool(node.children):
        return
    for child in node.children:
        if bool(node.children[child].children):
            post_pruning_recursion(tree, node.children[child], data)

    precision_without_prun = (tree.discriminate(data.iloc[:, :-1]) == data.iloc[:, -1]).sum() / data.shape[0]
    temp_children = node.children
    node.children = dict()
    precision_with_prun = (tree.discriminate(data.iloc[:, :-1]) == data.iloc[:, -1]).sum() / data.shape[0]
    if precision_without_prun - precision_with_prun < 0.03:
        node.attr = None
    else:
        node.children = temp_children


# Currently CART does not support missing values in training set.
def tree_generate_recursion(data=pd.DataFrame([]), type='ID3', continous_col=pd.Series([]), parent=None):
    node = TreeNode(parent)
    labelN = data.columns[-1]  # Name of the label column.
    A = data.columns[:-1]

    if (data.iloc[:, -1] == data.iloc[0, -1]).all(axis=None):
        node.set_label(data.iloc[0, -1])
        return node

    if A.empty or (data.iloc[:, :-1] == data.iloc[0, :-1]).all(axis=None):
        node.set_label(data.iloc[:, -1].value_counts(sort=False).idxmax())
        return node

    node.set_label(data.iloc[:, -1].value_counts(sort=False).idxmax())

    # Choose the optimal partition attribute.
    if type is 'ID3':
        gain = dict()  # attr -> its gain
        attr2t = dict()  # Continuous attr -> its optimal partition point.

        for attr in A:
            data_without_missing_on_attr = data.loc[data[attr].notnull(), [attr, labelN]]
            rowN = data_without_missing_on_attr.shape[0]
            ent_all = entropy(data_without_missing_on_attr.iloc[:, -1].value_counts(sort=False)) * np.log2(np.e)
            p = rowN / data.shape[0]  # p = notnull / all
            if attr in continous_col:
                sorted = data_without_missing_on_attr.sort_values(by=attr)  # ascending = True  by default

                if sorted[attr].unique().shape[0] == 1:
                    optml_t_of_attr = sorted[attr].unique()[0]
                    gain[attr] = 0
                    attr2t[attr] = optml_t_of_attr
                else:
                    Ta = pd.Series([(x + y) / 2 for x, y in zip(sorted[attr].unique()[:-1],
                                                                sorted[attr].unique()[1:])])
                    t2gain = dict()  # partition point t of a attr -> its gain
                    for t in Ta:
                        sorted['true_vec'] = (sorted[attr] > t)
                        ent_of_t_of_attr = np.sum(sorted.loc[:, ['true_vec', labelN]].groupby(by='true_vec').agg(
                            lambda x: np.size(x) * entropy(x.value_counts(sort=False)) * np.log2(np.e) / rowN))
                        t2gain[t] = (ent_all - ent_of_t_of_attr).iloc[0]
                    optml_t_of_attr = max(t2gain,
                                          key=t2gain.get)  # Return the optimal partition point optml_t_of_attr.
                    gain[attr] = p * t2gain[optml_t_of_attr]
                    attr2t[attr] = optml_t_of_attr
            else:
                ent_of_attr = np.sum(data_without_missing_on_attr.groupby(attr).agg(
                    lambda x: np.size(x) * entropy(x.value_counts(sort=False)) * np.log2(np.e) / rowN))
                gain[attr] = p * (ent_all - ent_of_attr).iloc[0]
        optml_attr = max(gain, key=gain.get)  # If same, return the first found, try to randomize!

    elif type is 'CART':
        ginis = dict()
        attr2t = dict()  # Continuous attr -> its optimal partition point.

        for attr in A:
            if attr in continous_col:
                sorted = data.loc[:, [attr, labelN]].sort_values(by=attr)  # ascending = True, by default
                Ta = pd.Series([(x + y) / 2 for x, y in zip(sorted.loc[::2, attr], sorted.loc[1::2, attr])])
                t2ginis = dict()  # partition point t of a attr -> its gain
                for t in Ta:
                    sorted['true_vec'] = (sorted[attr] > t)
                    ginis_of_t_of_attr = np.sum(sorted.loc[:, ['true_vec', labelN]].groupby(by='true_vec').agg(
                        lambda x: (1 - np.sum((x.value_counts(sort=False) / np.size(x)) ** 2)) * np.size(x) / rowN))
                    t2ginis[t] = ginis_of_t_of_attr.iloc[0]
                optml_t_of_attr = min(t2ginis,
                                      key=t2ginis.get)  # Return the optimal partition point optml_t_of_attr.
                ginis[attr] = t2ginis[optml_t_of_attr]
                attr2t[attr] = optml_t_of_attr
            else:
                gini = np.sum(data.loc[:, [attr, labelN]].groupby(attr).agg(
                    lambda x: (1 - np.sum((x.value_counts(sort=False) / np.size(x)) ** 2)) * np.size(x) / rowN))
                ginis[attr] = gini.iloc[0]
        optml_attr = min(ginis, key=ginis.get)

    node.attr = optml_attr
    Anew = list(A.copy())
    Anew.remove(optml_attr)

    if optml_attr in continous_col:
        node.is_continuous = attr2t[optml_attr]
        Dv = {True: data.loc[data[optml_attr] > attr2t[optml_attr]].index,
              False: data.loc[data[optml_attr] <= attr2t[optml_attr]].index}

        for tf in (True, False):
            if Dv[tf].empty:
                child_node = TreeNode()
                child_node.parent = node
                node.add_child(tf, child_node)
                node.children[tf].set_label(data.loc[:, labelN].value_counts(sort=False).idxmax())
            else:
                node.add_child(tf, tree_generate_recursion(data.loc[Dv[tf], Anew + [labelN]],
                                                           type=type, continous_col=continous_col, parent=node))
    else:
        attr_optml_value = data.loc[:, optml_attr].unique()  # A pd Series.
        for value in attr_optml_value:
            Dv = data.loc[data.loc[:, optml_attr] == value].index

            if Dv.empty:
                child_node = TreeNode()
                child_node.parent = node
                node.add_child(value, child_node)
                node.children[value].set_label(data.loc[:, labelN].value_counts(sort=False).idxmax())
            else:
                node.add_child(value, tree_generate_recursion(data.loc[Dv, Anew + [labelN]], type=type,
                                                              continous_col=continous_col, parent=node))

    # Generate weights i.e. estimated conditional prob.s  of each node.
    attr_val_without_missing = data.loc[data[optml_attr].notnull(), optml_attr]
    if optml_attr not in continous_col:
        val_count = attr_val_without_missing.value_counts(sort=False) / attr_val_without_missing.shape[0]
    else:
        val_count = (attr_val_without_missing > attr2t[optml_attr]).value_counts(sort=False) / \
                    attr_val_without_missing.shape[0]

    for idx in val_count.index:
        node.weights[idx] = val_count.loc[idx]

    return node
