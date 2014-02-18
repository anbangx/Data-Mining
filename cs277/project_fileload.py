import time
import cPickle as pickle

from nltk.stem.porter import *
import numpy as np
import pandas as pd

from cs277.DecisionTree.decisiontree import ClassificationTree
from sklearn import tree
import pydot
import StringIO

#
# Pre-Process Part
#

startTime = time.time()
outputFile = open('pre_processed_data_object', 'rb')

# File Fraction size to Read. Set between 0.1 and 1
fileFractionSize = pickle.load(outputFile)
fileTestFractionSize = pickle.load(outputFile)

# Define Regular Expression to pre-process strings. Only AlphaNumeric and whitespace will be kept.
strPattern = re.compile('[^a-zA-Z0-9 ]')

# A dictionary which keeps token and its frequency for each category. It will keep a Dictionary in a Dictionary.
# key - category, value-{'term':frequency}
# Example : {'acq' : {'hi':1,'compu':3,'move':1 ...}}
categoryAlphaNumericStrStemmedDict = pickle.load(outputFile)
categoryTestAlphaNumericStrStemmedDict = pickle.load(outputFile)

# A dictionary which keeps token, its frequency, and category for each file. It is layered Dictionary structure.
# 1st layer Dict {A}: key - category, value-{'term':frequency}
# 2nd layer Dict {B}: key - filename, value-{A}
# Example : {'000056' : {'acq' : {'hi':1, 'compu:3, 'move':1 ...}}}
fileAlphaNumericStrStemmedDict = pickle.load(outputFile)
fileTestAlphaNumericStrStemmedDict = pickle.load(outputFile)

# A list which keeps whole vocabularies throughout whole categories. It will be sorted.
# Example : ['current', 'curtail', 'custom', 'cut', 'cuurent', 'cvg', 'cwt', 'cypru', 'cyrpu', 'd', 'daili' ...]
wholeVocabularyList = pickle.load(outputFile)
wholeTestVocabularyList = pickle.load(outputFile)

wholeVocabularyFrequency = pickle.load(outputFile)
wholeTestVocabularyFrequency = pickle.load(outputFile)

# A dictionary which keeps entire vocabulary and its frequency across whole categories
# Example : {'current' : 110, 'said' : 10000 ...... }
wholeVocabularyFrequencyDict = pickle.load(outputFile)
wholeVocabularyTestFrequencyDict = pickle.load(outputFile)

# A dictionary which keeps number of files in each category
# Example : {'acq': 115, 'alum': 222 ...}
numberOfFilesInEachCategoryDict = pickle.load(outputFile)
numberOfFilesInEachCategoryTestDict = pickle.load(outputFile) 

# A dictionary which keeps fraction of [number of files in each category] / [number of entire files]
# Example : {'acq':0.015, 'alum':0.031 ...}
fractionOfFilesInEachCategoryDict = pickle.load(outputFile) 
fractionOfFilesInEachCategoryTestDict = pickle.load(outputFile) 


categoryNum = pickle.load(outputFile)
fileNum = pickle.load(outputFile)

categoryTestNum = pickle.load(outputFile)
fileTestNum = pickle.load(outputFile)


# A two dimensional List which keeps frequency of term per category. 
# row = category. column = frequency of each term in that category.
# For term list, we are using whole terms across entire categories.
# Example : category- acq, bop, term- 'commonplac', 'commonwealth', 'commun'
#           commonplac   commonwealth  commun
#    acq         7              2         0
#    bop         8              9         1 
termFrequencyPerCategoryList = pickle.load(outputFile)

print str(time.time() - startTime)
print
# print categoryNum
# print fileNum
# print categoryTestNum
# print fileTestNum

# print wholeVocabularyTestFrequencyDict

# print len(wholeVocabularyList)
# print len(wholeTestVocabularyList)
#
# print wholeVocabularyFrequency
# print wholeTestVocabularyFrequency
#
# print numberOfFilesInEachCategoryDict['austdlr']

print termFrequencyPerCategoryList

# Define TF-IDF based Cosine Similarity algorithm    
def tfidfCosineSimilarity(list):
    print "\nTF-IDF Cosine Similarity Algorithm\n"

# Define TF-IDF based Cosine Similarity algorithm in Detail    
def tfidfCosineSimilarityDetail(list):
    print "\nTF-IDF Cosine Similarity Algorithm\n"

# Define Decision Tree algorithm.
def decisionTree(training_dict, testing_dict, use_sklearn_lib=False):
    print "\nDecision Tree Algorithm\n"
    if use_sklearn_lib:
        decisionTree_sklearn(training_dict, testing_dict)
    else:
        decisionTree_own_version(training_dict)

def decisionTree_sklearn(trainning_dict, testing_dict, criterion='entropy', max_depth=10, draw=False):
    print "\nUsing sklearn library.... \n"
    print trainning_dict
    # Example : {'000056' : {'acq' : {'hi':1, 'compu:3, 'move':1 ...}}}
    X_train = []
    Y_train = []
    for termFreq_per_category in trainning_dict.values():
        for category, termFreq in termFreq_per_category.items():
            X_train.append(termFreq)
            Y_train.append(category)

    X_test = []
    Y_test = []
    for termFreq_per_category in testing_dict.values():
        for category, termFreq in termFreq_per_category.items():
            X_test.append(termFreq)
            Y_test.append(category)

    clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    clf = clf.fit(X_train, Y_train)
    # predict and compute correct rate
    correct = 0
    for i in range(len(X_test)):
        predict = clf.predict(X_test[i])
        print predict
        if predict == Y_test[i]:
            correct += 1
    correct_rate = correct / len(X_test)
    print correct_rate

    if draw:
        dot_data = StringIO.StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("DecisionTree.pdf")

def decisionTree_own_version(list, num_categories=5, num_features=1000, min_node_size=2, max_node_depth=10):
    list = np.array(list)
    list = list[:num_categories]
    list = list[:, 0:num_features]
    print list

    columns = []
    for i in range(len(list[0]) - 1):
        print str(i)
        columns.append(str(i))
    columns.insert(0, "category")
    df = pd.DataFrame(list, columns=columns)

    g = ClassificationTree()
    parameters = dict()
    parameters['min_node_size'] = min_node_size
    parameters['max_node_depth'] = max_node_depth
    parameters['threshold'] = 0
    parameters['response'] = 'category'
    parameters['alpha'] = 0
    parameters['metric_kind'] = 'Entropy'
    g.train(data=df, parameters=parameters)
    g.plot()

# Define Decision Tree Algorithm in detail
def decisionTreeDetail(list):
    print "\nDecision Tree Algorithm\n"

# Define Naive Bayes algorithm
def naiveBayes(list):
    print "\nNaive Bayes Algorithm\n"

# Define Naive Bayes algorithm in detail
def naiveBayesDetail(list):
    print "\nNaive Bayes Algorithm\n"

# Execute TF-IDF based Cosine Similarity algorithm    
tfidfCosineSimilarity(termFrequencyPerCategoryList)

# Execute Decision Tree algorithm
decisionTree(termFrequencyPerCategoryList, fileTestAlphaNumericStrStemmedDict)

# Execute NaiveBayes algorithm
naiveBayes(termFrequencyPerCategoryList)
    

