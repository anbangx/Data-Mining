__author__ = 'anbangx'

from sklearn import tree

X = [[0, 0], [1, 1], [2, 2]]
Y = [0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
pred = clf.predict([[0., 0.]])
print pred