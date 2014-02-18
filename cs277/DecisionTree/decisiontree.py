__author__ = 'anbangx'

import graphs as tig
import igraph as ig
from math import log

class DecisionNode(tig.BasicNode, object):
    def __init__(self, **kwargs):
        super(DecisionNode, self).__init__(**kwargs)
        self.left = None
        self.right = None
        self.children = None
        self.parent = None
        self.prediction = None
        self.predicted_class = None
        self.tally = {}
        self.total = 0.0
        self.size = 0
        self.depth = 0
        self.local_data = None
        self.error = None
        
    def local_filter(self, data): #filters data
        pass

    def get_next_node(self, datapoint): 
        pass

class DecisionTree(tig.return_nary_tree_class(directed=True), object):
    def __init__(self, Vertex=DecisionNode, **kwargs):
        super(DecisionTree, self).__init__(N=2, Vertex=Vertex, **kwargs)
        self.data = None
        self.data_size = 0
        self.response = '' #data attribute we're trying to predict
        self.metric_kind = ''

    def split_vertex(self, vertex):
        super(DecisionTree, self).split_vertex(vertex)
        vertex.left = vertex.children[0]
        vertex.left.depth = vertex.depth+1
        vertex.right = vertex.children[1]
        vertex.right.depth = vertex.depth+1

    def fuse_vertex(self, vertex):
        super(DecisionTree, self).fuse_vertex(vertex)
        vertex.left, vertex.right = None, None
        
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

class PivotDecisionNode(DecisionNode, object):
    def __init__(self, **kwargs):
        super(PivotDecisionNode, self).__init__(**kwargs)
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
     
    def get_data_leaf(self, datapoint):
        if self.children is None:
            return self
        else:
            if datapoint[self.split_attribute] <= self.pivot:
                return self.left.get_data_leaf(datapoint)
            else:
                return self.right.get_data_leaf(datapoint)
        
class PivotDecisionTree(DecisionTree, object):
    # def __init__(self, data, metric_kind, max_node_depth, Vertex=PivotDecisionNode, **kwargs):
    def __init__(self, Vertex=PivotDecisionNode, **kwargs):
        super(PivotDecisionTree, self).__init__(Vertex=Vertex, **kwargs)
        #these are default, can be set by train
        self.min_node_size = 0
        self.max_node_depth = 5 # max_node_depth;
        self.threshold = 0
        # self.data = data
        # self.metric_kind = metric_kind

    def split_vertex(self, vertex, split_attribute, pivot):
        super(PivotDecisionTree, self).split_vertex(vertex)
        vertex.pivot, vertex.split_attribute = pivot, split_attribute
        
    def fuse_vertex(self, vertex):
        super(PivotDecisionTree, self).fuse_vertex(vertex)
        vertex.pivot, vertex.split_attribute = None, None

    def grow_tree(self):
        self.data_size = len(self.data)
        self.create_vertex()
        self.set_root(self.vertices[0])
        self.leaves.add(self.vertices[0])
        self.grow_node(node=self.get_root())
        self.set_predictions()

    def grow_node(self, node):
        if node.parent is None:
            node.local_data = node.local_filter(data=self.data)
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
                child.local_data=child.local_filter(data=node.local_data)
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
        for split in self.iter_split(node):
            if node.children is None:
                pass
            else:
                for child in node.children:
                    child.local_data=child.local_filter(node.local_data)
                ret = [self.node_purity(node),
                node.split_attribute, node.pivot]
                yield ret
         
    def iter_split(self, node):
        for attribute in self.data.columns:
            if attribute != self.response:
                for pivot in self.get_pivots(node.local_data, attribute):
                    self.fuse_vertex(node)
                    self.split_vertex(vertex=node, pivot=pivot, 
                                      split_attribute=attribute)
                    yield

    def get_pivots(self, data, attribute):
        max_pivot = max(data[attribute].unique())
        for pivot in data[attribute].unique():
            if pivot < max_pivot:
                yield pivot

    def set_predictions(self):
        for node in self.vertices:
            self.set_node_prediction(node)

    def stopping_condition(self, node):
        if self.max_node_depth <= node.depth:
            return True
        elif node.size <= self.min_node_size:
            return True
        else:
            return False

    def train(self, data, parameters, prune=False):
        self.vertices = []
        self.edges = set([])
        self.leaves = set([])
        self.data = data
        self.response = parameters['response']
        self.metric_kind = parameters['metric_kind']
        self.min_node_size = parameters['min_node_size']
        self.max_node_depth = parameters['max_node_depth']
        self.threshold = parameters['threshold']
        alpha = parameters['alpha']
        self.grow_tree()
        if prune:
            self.prune_tree(alpha)

    def predict(self, data_point, class_probs=False):
        return self.vertices[0].get_data_leaf(data_point).prediction
        
    def test(self, data):
        self.load_new_data(data)
        return self.error(new_data=True)

class ClassificationTree(PivotDecisionTree, object):
    def __init__(self, **kwargs):
        super(ClassificationTree, self).__init__(**kwargs)
        #these are default, can be set by train
        self.metric_kind = 'Entropy'

    def set_node_prediction(self, node):
        node.prediction = node.local_data[self.response].value_counts()
        node.size = sum(node.prediction[key] for key in node.prediction.keys())
        node.size = float(node.size)
        node.prediction = {key: node.prediction[key]/node.size
                          for key in node.prediction.keys()}

        key, value = max(node.prediction.iteritems(), key=lambda x:x[1])
        node.predicted_class = key
        node.predicted_prob = value

    def plot(self, margin=50):
        A = self.get_adjacency_matrix_as_list()
        convert_to_igraph = ig.Graph.Adjacency(A)
        g=convert_to_igraph
        for vertex in self.vertices:
            index = self.vertices.index(vertex)
            if vertex.pivot is not None:

                 label_pivot = ' <= '+str(vertex.pivot)
                 g.vs[index]['label'] = 'w' + str(vertex.split_attribute) + label_pivot
                 g.vs[index]['label_dist'] = 2
                 g.vs[index]['label_color'] = 'red'
                 g.vs[index]['color'] = 'red'

            else:
                label = str(vertex.predicted_class)
                g.vs[index]['color'] = 'blue'
                g.vs[index]['label'] = label

                g.vs[index]['label_dist'] = 2
                g.vs[index]['label_color'] = 'blue'
        root_index = self.vertices.index(self.get_root())
        layout = g.layout_reingold_tilford(root=root_index)
        ig.plot(g, layout=layout, margin=margin)

    def predict(self, data_point, class_probs=False):
        if class_probs:
            return self.vertices[0].get_data_leaf(data_point).prediction
        else:
            return self.vertices[0].get_data_leaf(data_point).predicted_class