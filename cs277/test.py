__author__ = 'anbangx'

from sklearn import tree
import collections

X = [[0, 0, 3], [1, 1, 2], [2, 2, 1], [3, 3, 3], [4, 0, 2], [5, 1, 1]]
Y = [0, 1, 2, 2, 1, 0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

predict = clf.predict_proba([[1, 1, 1], [2, 2, 2], [1, 2, 3], [3, 2, 1]])
print str(predict)

def get_first_k_categories(value_list, name_list, k=1):
    value_to_name_dict = dict(zip(value_list, name_list))
    ordered_dict = collections.OrderedDict(sorted(value_to_name_dict.items()))
    first_k = []
    for i in range(k):
        print str(ordered_dict[k])
        first_k.append(ordered_dict[k])
    print first_k