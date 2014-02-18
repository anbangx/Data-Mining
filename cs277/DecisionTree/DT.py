__author__ = 'anbangx'

from math import log
import numpy as np
import pandas as pd

def dict_to_list(dict):
    list = []
    for k, v in dict.items():
        v.insert(0, k)
        list.append(v)
    return list

trainning_dict = {}
list1 = [1, 5, 3, 1, 4]
list2 = [1, 7, 3, 1, 4]
list3 = [3, 5, 3, 1, 4]
list4 = [3, 7, 3, 1, 4]
trainning_dict['c1'] = list1
trainning_dict['c2'] = list2
trainning_dict['c3'] = list3
trainning_dict['c4'] = list4
list = np.array(dict_to_list(trainning_dict))
print list

columns = []
for i in range(len(list[0]) - 1):
    print str(i)
    columns.append(str(i))
columns.insert(0, "category")
df = pd.DataFrame(list, columns=columns)

global_data = df

class PivotDecisionNode():
    def __init__(self):
        self.local_data = None
        self.size = 0
        self.left = None
        self.right = None
        self.children = None
        self.parent = None
        self.depth = 0
        self.pivot = None
        self.split_attribute = None

    def local_filter(self, data):
        if self.parent is None:
            self.size = len(data)
            return data
        attribute = self.parent.split_attribute
        pivot = self.parent.pivot

        ret = data[attribute] <= pivot
        if self == self.parent.left:
            ret = data[ret]
        else:
            ret = data[~ret]
        self.size = len(ret)
        return ret

class PivotDecisionTree():
    def __init__(self):
        #these are default, can be set by train
        print 'in PivotDecisionTree'
        self.root = None
        self.local_data = None
        self.nodes = set()
        self.response = 'category' # data attribute we're trying to predict
        self.metric_kind = 'entropy'
        self.max_node_depth = 5
        self.min_node_size = 1

    def create_vertex(self):
        new_vertex = PivotDecisionNode()
        self.nodes.add(new_vertex)
        return new_vertex

    def grow_tree(self):
        self.root = self.create_vertex()
        self.grow_node(node=self.root)
        self.set_predictions()

    def set_predictions(self):
        for node in self.nodes:
            self.set_node_prediction(node)

    def set_node_prediction(self, node):
        node.prediction = node.local_data[self.response].value_counts()
        node.size = sum(node.prediction[key] for key in node.prediction.keys())
        node.size = float(node.size)
        node.prediction = {key: node.prediction[key]/node.size for key in node.prediction.keys()}

    def grow_node(self, node):
        if node.parent is None:
            global global_data
            node.local_data = node.local_filter(data=global_data)
        if self.stopping_condition(node):
            return
        else:
            try:
                best_split = self.get_best_split(node)

            except StopIteration:
                return
            self.split_vertex(node, split_attribute=best_split[1],
                              pivot=best_split[2])
            for child in node.children:
                child.local_data = child.local_filter(data=node.local_data)
                self.grow_node(node=child)

    def get_best_split(self, node):
        gen = self.iter_split_eval(node)
        first = next(gen)
        best_split = first
        for split in gen:
            if split[0] < best_split[0]:
                best_split = split
        return best_split

    def iter_split_eval(self, node):
        for attribute in self.iter_split(node):
            if node.children is None:
                pass
            else:
                for child in node.children:
                    child.local_data = child.local_filter(node.local_data)
                ret = [self.node_purity(node), node.split_attribute, node.pivot]
                yield ret

    def iter_split(self, node):
        for attribute in node.local_data.columns:
            if attribute != self.response:
                for pivot in self.get_pivots(node.local_data, attribute):
                    self.fuse_vertex(node)
                    self.split_vertex(vertex=node, pivot=pivot, split_attribute=attribute)
                    yield attribute

    def get_pivots(self, data, attribute):
        max_pivot = max(data[attribute].unique())
        for pivot in data[attribute].unique():
            if pivot < max_pivot:
                yield pivot

    def stopping_condition(self, node):
        if self.max_node_depth <= node.depth:
            return True
        elif node.size <= self.min_node_size:
            return True
        else:
            return False

    def split_vertex(self, vertex, split_attribute, pivot):
        children = [self.create_vertex() for i in range(2)]
        vertex.children = children
        for Child in children:
            Child.parent = vertex
        vertex.left = vertex.children[0]
        vertex.left.depth = vertex.depth+1
        vertex.right = vertex.children[1]
        vertex.right.depth = vertex.depth+1
        vertex.pivot, vertex.split_attribute = pivot, split_attribute

    def fuse_vertex(self, vertex):
        self.nodes.remove(vertex.children)
        vertex.children = None
        vertex.left, vertex.right = None, None
        vertex.pivot, vertex.split_attribute = None, None

    def node_purity(self, node):
        if node.children is None:
            return self.metric(node.local_data, kind=self.metric_kind)
        else:
            left_raw_purity = self.node_purity(node=node.left)
            right_raw_purity = self.node_purity(node=node.right)
            left_size = float(node.left.size)
            right_size = float(node.right.size)
            left_purity = (left_size/node.size)*left_raw_purity
            right_purity = (right_size/node.size)*right_raw_purity
            return left_purity+right_purity

    def Gini(self, prob_vector):
        return sum(p*(1-p) for p in prob_vector)

    def Entropy(self, prob_vector):
        def entropy_summand(p):
            if p == 0:
                return 0
            else:

                return -p*log(p,2)
        return sum(entropy_summand(p) for p in prob_vector)

    def metric(self, filtered_data, kind):
        prob_vector = self.get_prob_vector(filtered_data)
        if kind == 'Entropy':
            return self.Entropy(prob_vector)
        elif kind == 'Gini':
            return self.Entropy(prob_vector)

    def get_prob_vector(self, data):
        size = float(len(data))
        value_count = data[self.response].value_counts()
        prob_vector = [value_count[key]/size for key in value_count.keys()]
        return prob_vector

class ClassificationTree(PivotDecisionTree):
    def __init__(self):
        PivotDecisionTree.__init__(self)
        print 'in ClassificationTree'

    def train(self, parameters):
        self.response = parameters['response']
        self.metric_kind = parameters['metric_kind']
        self.max_node_depth = parameters['max_node_depth']
        self.min_node_size = parameters['min_node_size']
        self.grow_tree()

    def set_node_prediction(self, node):
        node.prediction = node.local_data[self.response].value_counts()
        node.size = sum(node.prediction[key] for key in node.prediction.keys())
        node.size = float(node.size)
        node.prediction = {key: node.prediction[key]/node.size for key in node.prediction.keys()}

        key, value = max(node.prediction.iteritems(), key=lambda x: x[1])
        node.predicted_class = key
        node.predicted_prob = value

g = ClassificationTree()
parameters = dict()
parameters['response'] = 'category'
parameters['metric_kind'] = 'Entropy'
parameters['min_node_size'] = 1
parameters['max_node_depth'] = 5
g.train(parameters=parameters)