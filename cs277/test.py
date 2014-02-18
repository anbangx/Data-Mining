__author__ = 'anbangx'

from sklearn import tree

def dict_to_list(dict):
    list = []
    for k, v in dict.items():
        v.append(k)
        list.append(v)
    return list

# dict = {}
# list1 = [1, 2, 3, 4, 5]
# list2 = [5, 2, 3, 1, 4]
# dict['c1'] = list1
# dict['c2'] = list2
# res = dict_to_list(dict)
# print res
#
# X = [[0, 0], [1, 1], [2, 2]]
# Y = [0, 1, 1]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, Y)
# pred = clf.predict([[0., 0.]])
# print pred
