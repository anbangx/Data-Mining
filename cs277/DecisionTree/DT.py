__author__ = 'anbangx'

from math import log
import Queue
from sklearn import tree
import pandas as pd
import numpy as np
import time
import pydot
import StringIO

global_data = ''
def set_global_data(data):
    global global_data
    global_data = data

class PivotDecisionNode():
    def __init__(self):
        self.local_data = None
        self.size = 0
        self.left = None
        self.right = None
        self.children = None
        self.parent = None
        self.depth = 0
        self.split_attribute = None
        self.pivot = None
        self.prediction = None
        self.predicted_class = None

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

    def get_data_leaf(self, datapoint):
        if self.children is None:
            return self
        else:
            # print datapoint[self.split_attribute]
            if datapoint[self.split_attribute][0] <= self.pivot:
                return self.left.get_data_leaf(datapoint)
            else:
                return self.right.get_data_leaf(datapoint)

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
        if node.local_data is not None and len(node.local_data) != 0:
            node.prediction = node.local_data[self.response].value_counts()
            node.size = sum(node.prediction[key] for key in node.prediction.keys())
            node.size = float(node.size)
            node.prediction = {key: node.prediction[key]/node.size for key in node.prediction.keys()}

            key, value = max(node.prediction.iteritems(), key=lambda x: x[1])
            node.predicted_class = key
            node.predicted_prob = value

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
            self.fuse_vertex(node)
            self.split_vertex(node, split_attribute=best_split[1],
                              pivot=best_split[2])
            node.left.local_data = node.left.local_filter(data=node.local_data)
            node.right.local_data = node.right.local_filter(data=node.local_data)
            node.children[0] = node.left
            node.children[1] = node.right
            del node.local_data
            node.local_data = None
            self.grow_node(node=node.left)
            self.grow_node(node=node.right)

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
                node.left.local_data = node.left.local_filter(data=node.local_data)
                node.right.local_data = node.right.local_filter(data=node.local_data)
                ret = [self.node_purity(node), node.split_attribute, node.pivot]
                node.left.local_data = None
                node.right.local_data = None
                del node.children
                node.left = None
                node.right = None
                node.children = None
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
        if vertex.children is None:
            return
        for child in vertex.children:
            self.nodes.remove(child)
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

    def metric(self, filtered_data, kind):
        prob_vector = self.get_prob_vector(filtered_data)
        if kind == 'Entropy':
            return self.Entropy(prob_vector)
        elif kind == 'Gini':
            return self.Entropy(prob_vector)

    def Gini(self, prob_vector):
        return sum(p*(1-p) for p in prob_vector)

    def Entropy(self, prob_vector):
        def entropy_summand(p):
            if p == 0:
                return 0
            else:

                return -p*log(p,2)
        return sum(entropy_summand(p) for p in prob_vector)

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

    def predict(self, data_point, class_probs=False):
        if class_probs:
            return self.root.get_data_leaf(data_point).prediction
        else:
            return self.root.get_data_leaf(data_point).predicted_class

    def plot(self):
        f = open('./tree.output', 'w')

        q = Queue.Queue()
        q.put(self.root)
        while not q.empty():
            cur_node = q.get()
            f.write(self.seriablize(cur_node) + '\n')
            if cur_node.left != None:
                q.put(cur_node.left)
            if cur_node.right != None:
                q.put(cur_node.right)
        f.close()

    def seriablize(self, node):
        s = ''
        s += 'depth:' + str(node.depth) + ' '
        if node.split_attribute is not None:
            s += 'split_attribute: ' + str(node.split_attribute) + ' '
        if node.pivot is not None:
            s += 'frequence_pivot: ' + str(node.pivot) + ' '
        if node.prediction is not None:
            s += 'prediction: ' + str(node.prediction)
        # if node.left is None and node.right is None: # if leaf, print prob
        return s

def dict_to_list(dict):
    list = []
    for k, v in dict.items():
        v.insert(0, k)
        list.append(v)
    return list

debug = False
if debug:
    trainning_dict = {}
    list1 = [1, 5, 3, 1, 4]
    list2 = [1, 7, 3, 1, 4]
    list3 = [3, 5, 3, 1, 4]
    list4 = [3, 7, 3, 1, 4]
    list5 = [3, 8, 3, 1, 4]
    list6 = [3, 8, 3, 1, 7]
    list7 = [3, 8, 3, 1, 7]

    trainning_dict['c1'] = list1
    trainning_dict['c2'] = list2
    trainning_dict['c3'] = list3
    trainning_dict['c4'] = list4
    trainning_dict['c5'] = list5
    list = dict_to_list(trainning_dict)
    list.append(['c4', 3, 7, 3, 1, 4])
    list = np.array(list)
    print list
    #
    columns = []
    for i in range(len(list[0]) - 1):
        columns.append('w' + str(i + 1))
        columns.insert(0, "category")
    df = pd.DataFrame(list, columns=columns)
    print '-------'
    print df['w1']
    global_data = df
    g = ClassificationTree()
    parameters = dict()
    parameters['response'] = 'category'
    parameters['metric_kind'] = 'Entropy'
    parameters['min_node_size'] = 1
    parameters['max_node_depth'] = 5
    g.train(parameters=parameters)
    g.plot()

    list1 = [1, 5, 3, 1, 4]
    columns = ['w1', 'w2', 'w3', 'w4', 'w5']
    # print columns
    datapoint = pd.DataFrame(np.array([list1]), columns=columns)
    predict = g.predict(datapoint)
    print 'The prediction of ' + str(list1) + ' is ' + predict

def decisionTree_version1(trainning_list, testing_list, words_name, num_trainning_file=100, num_features=2000, min_node_size=1, max_node_depth=4, plot=True):
    print "\nUsing decision tree..... \n"
    trainning_start_time = time.time()
    trainning_list = trainning_list[:num_trainning_file]
    trainning_features = [row[0:num_features] for row in trainning_list]
    trainning_prediction_class = [row[-1] for row in trainning_list]
    for i in range(len(trainning_features) - 1):
        trainning_features[i].append(trainning_prediction_class[i]) # including prediction_class
    trainning_list = trainning_features
    words_name = words_name[0:num_features]
    words_name.append("category")
    df = pd.DataFrame(trainning_list, columns=words_name)

    g = ClassificationTree()
    set_global_data(df)
    parameters = dict()
    parameters['min_node_size'] = min_node_size
    parameters['max_node_depth'] = 3; max_node_depth
    parameters['response'] = 'category'
    parameters['metric_kind'] = 'Entropy'
    g.train(parameters=parameters)
    if plot:
        g.plot()
    print 'Execution Time (sec) -  Training the Data Set: ' + str(time.time() - trainning_start_time)

    # testing_start_time = time.time()
    # # predict and compute correct rate
    # words_name = words_name[0:num_features]
    # num_correct = 0
    # total = len(testing_list)
    # for file in testing_list:
    #     datapoint = pd.DataFrame(np.array([file[:num_features]]), columns=words_name)
    #     predict = g.predict(datapoint)
    #     if predict == file[-1]:
    #         num_correct += 1
    # print 'The number of correct prediction is: ' + str(num_correct) + ' and the total number is: ' + str(total)
    # print 'The correctness is ' + str(float(num_correct)/total)
    #
    # print 'Execution Time (sec) - Testing the Data Set: ' + str(time.time() - testing_start_time)
    # print 'Execution Time (sec) - Overall (Training + Test): ' + str(time.time() - trainning_start_time)
    # print '\nFinished decision tree.....'

def decisionTree_version2(trainning_list, testing_list, fileTestBelongCategory, criterion='gini', max_depth=None, adjust_depth_dict=None, draw=False):
    # correctness_and_time = []
    # adjust_depth_dict[max_depth] = correctness_and_time
    print "\nUsing decision tree..... \n"
    trainning_start_time = time.time()
    num_features = len(trainning_list[0])-1
    X_train = [row[0:num_features - 1] for row in trainning_list] #row[0:num_features - 1]
    Y_train = [row[-1] for row in trainning_list]

    if max_depth is None:
        clf = tree.DecisionTreeClassifier(criterion=criterion) # , max_features='log2'
    else:
        clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features='sqrt')
    clf = clf.fit(X_train, Y_train)
    print 'Execution Time (sec) -  Training the Data Set: ' + str(time.time() - trainning_start_time)

    testing_start_time = time.time()
    # predict and compute correct rate
    num_correct = 0
    total = len(testing_list)
    for file in testing_list:
        predict = clf.predict(file[0:num_features - 1])
        if predict in fileTestBelongCategory[file[-1]]:
            num_correct += 1
    print 'The number of correct prediction is: ' + str(num_correct) + ' and the total number is: ' + str(total)
    correctness = float(num_correct)/total
    print 'The correctness is ' + str(correctness)

    print 'Execution Time (sec) - Testing the Data Set: ' + str(time.time() - testing_start_time)
    print 'Execution Time (sec) - Overall (Training + Test): ' + str(time.time() - trainning_start_time)
    print '\nFinished decision tree.....'

    # correctness_and_time.append(correctness)
    # correctness_and_time.append(time.time() - trainning_start_time)

    if draw:
        dot_data = StringIO.StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("DecisionTree.pdf")