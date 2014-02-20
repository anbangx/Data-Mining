__author__ = 'anbangx'

from sklearn import tree
X = [[0, 0], [1, 1]]
Y = ['first', 'second']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)