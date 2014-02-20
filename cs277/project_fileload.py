import os
import re
from nltk.corpus import stopwords
from nltk.stem.porter import *
import collections
import numpy as np
import time
import cPickle as pickle
import copy
import math
import pandas as pd
import cs277.DecisionTree.DT as DT
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

# A dictionary which keeps test filename, and its categories in Set
# {'000056' : ('acq', 'alum')}
fileTestBelongCategory = pickle.load(outputFile)

# print fileTestBelongCategory

# For entire vocabularies in the training set, create a dictionary that a list (value) which contains frequency per category (key)
# Orders of vocabularies are same for every list. The order is as same as that of in wholeVocabularyFromTrainingAndTestSetList.
# Example : { 'category' : '[frequency for 'said', frequency for 'mln' ...]', 'category' : '[frequency for 'said', frequency for 'mln' ...]'  
normalizedFrequencyPerCategoryInTrainingSetDict = pickle.load(outputFile)

def return_normalizedDict():
    return normalizedFrequencyPerCategoryInTrainingSetDict

# For entire vocabularies in the test set, create a dictionary that a list (value) which contains frequency per file (key)
# Orders of vocabularies are same for every list. The order is as same as that of in wholeVocabularyFromTrainingAndTestSetList.
# Example : { '0001268' : '[frequency for 'said', frequency for 'mln' ...]', 'category' : '[frequency for 'said', frequency for 'mln' ...]'   
normalizedFrequencyPerTestFileDict = pickle.load(outputFile)

# Entire Vocubulary List which include every terms from the training set and test set.
wholeVocabularyFromTrainingAndTestSetList = pickle.load(outputFile)

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

print "Object loading finished. Elapsed Time: " + str(time.time() - startTime)
print
print categoryNum
print fileNum
print categoryTestNum
print fileTestNum

# print len(fileTestAlphaNumericStrStemmedDict)

# print wholeVocabularyTestFrequencyDict

# print len(wholeVocabularyList)
# print len(wholeTestVocabularyList)
# 
# print wholeVocabularyFrequency
# print wholeTestVocabularyFrequency
# 
# print numberOfFilesInEachCategoryDict['austdlr']

# A dictionary which keeps Normalized TF, IDF, TF * IDF per category
# categoryAlphaNumericStrStemmedNormalizedTFDict = { 'category' : {'term' : 'normalized frequency', 'term' : ...}}
# wholeVocabularyIDFDict = { 'term' : 'IDF', 'term' : 'IDF' ...}}
# categoryAlphaNumericStrStemmedTFIDFDict = { 'category' : {'term' : 'normalized frequency * it's IDF', 'term' : ...}}
# categoryAlphaNumericStrStemmedTFIDFDict = { 'category' : {'term' : 'TF * IDF is normalized by vector length', 'term' : ...}}
categoryAlphaNumericStrStemmedNormalizedTFDict = {}
wholeVocabularyIDFDict = {}
categoryAlphaNumericStrStemmedTFIDFDict = {}
categoryAlphaNumericStrStemmedTFIDFUnitVectorDict = {}

fileTestAlphaNumericStrStemmedNormalizedTFDict = {}
fileTestAlphaNumericStrStemmedNormalizedTFUnitVectorDict = {}

fileTestCosineDistancePerCategory = {}

# Define TF-IDF based Cosine Similarity algorithm    
def tfidfCosineSimilarity(list):
    global categoryAlphaNumericStrStemmedDict, categoryAlphaNumericStrStemmedNormalizedTFDict, wholeVocabularyIDFDict
    totalFrequencyCountPerCategory = {}
    
    # Calculating Normalized TF
    # key:category, value:{term:frequency ...}
    for key, value in categoryAlphaNumericStrStemmedDict.iteritems():
    
        tmpTotalCountPerCategory = 0
        tmpNormalizedFrequencyPerCategory = {}
            
        # key1:term, value1:frequency
        for key1, value1 in value.iteritems():
            tmpTotalCountPerCategory += value1
            
        totalFrequencyCountPerCategory[key] = tmpTotalCountPerCategory
        
        # Put Normalized Frequency
        for key1, value1 in value.iteritems():
            tmpNormalizedFrequencyPerCategory[key1] = float(value1) / float(tmpTotalCountPerCategory)
            
        categoryAlphaNumericStrStemmedNormalizedTFDict[key] = tmpNormalizedFrequencyPerCategory

    # Calculating Inversed Document Frequency (IDF)
    # key: keyword
    for key in wholeVocabularyList:
        tmpTotalCountPerCategory = 0
        
        # key1:category, value1:{term:frequency ...}
        for key1, value1 in categoryAlphaNumericStrStemmedDict.iteritems():
            
            # If keyword is found in a category:
            if key in value1:
                tmpTotalCountPerCategory += 1

        wholeVocabularyIDFDict[key] = 1 + math.log(float(categoryNum) / float(tmpTotalCountPerCategory))
    
    # Calculate TF * IDF Score for each term
    # and Make TF*IDF Vector to Unit Vector
    for key,value in categoryAlphaNumericStrStemmedNormalizedTFDict.iteritems():
        
        tmpTFIDFDictPerCategory = {}
        tmpTFIDFUnitVectorDictPerCategory = {}
        tmpTFIDFDistance = 0
        
        for key1, value1 in value.iteritems():
            tmp = value1 * wholeVocabularyIDFDict[key1]
            tmpTFIDFDictPerCategory[key1] = tmp
            tmpTFIDFDistance += tmp * tmp
            
        categoryAlphaNumericStrStemmedTFIDFDict[key] = tmpTFIDFDictPerCategory
    
        for key1, value1 in value.iteritems():
            tmpTFIDFUnitVectorDictPerCategory[key1] = tmpTFIDFDictPerCategory[key1] / math.sqrt(float(tmpTFIDFDistance))
            
        categoryAlphaNumericStrStemmedTFIDFUnitVectorDict[key] = tmpTFIDFUnitVectorDictPerCategory
   
#     for key1, value1 in categoryAlphaNumericStrStemmedTFIDFUnitVectorDict['livestock'].iteritems():
#         print key1 + "\t" + str(value1)

   
    # Now, Calculate Normalized TF for each TEST document
    # key:filename, value : {'category' : {'term':frequency ... }}
    for key, value in fileTestAlphaNumericStrStemmedDict.iteritems():
        
        #  key1: category, value1: {term:frequency ...}
        for key1, value1 in value.iteritems():
            
            tmpTotalCountPerCategory = 0
            tmpNormalizedFrequencyPerCategory = {}
            tmpTFIDFUnitVectorDictPerCategory = {}
            tmpTFIDFDistance = 0.0
            
            # Calculate Total Frequency
            # key2: term, value2: frequency
            for key2, value2 in value1.iteritems():
                tmpTotalCountPerCategory += value2
            
            for key2, value2 in value1.iteritems():
                tmp = float(value2) / float(tmpTotalCountPerCategory)
                tmpNormalizedFrequencyPerCategory[key2] = tmp
                tmpTFIDFDistance += tmp * tmp
                
            for key2, value2 in value1.iteritems():
                tmpTFIDFUnitVectorDictPerCategory[key2] = tmpNormalizedFrequencyPerCategory[key2] / math.sqrt(float(tmpTFIDFDistance))
                
        fileTestAlphaNumericStrStemmedNormalizedTFDict[key] = tmpNormalizedFrequencyPerCategory
        fileTestAlphaNumericStrStemmedNormalizedTFUnitVectorDict[key] = tmpTFIDFUnitVectorDictPerCategory
        
#     for key1, value1 in fileTestAlphaNumericStrStemmedNormalizedTFUnitVectorDict['0009701'].iteritems():
#         print key1 + "\t" + str(value1)

    # Calculate Cosine Distance For each test file VS each category
    count = 0
    
    # key : test file name, value : { term : TF in Unit Vector ... }
    for key, value in fileTestAlphaNumericStrStemmedNormalizedTFUnitVectorDict.iteritems():
        
        tmpCosineDistancePerCategory = {}
            
        # key1 : term, value1 : TF in Unit vector
        for key1, value1 in value.iteritems():
                
            # key2 : category, value2: {term, TF*IDF in Unit Vector}
            for key2, value2 in categoryAlphaNumericStrStemmedTFIDFUnitVectorDict.iteritems():
                
                # Found 
                if key1 in value2:
                    if key2 in tmpCosineDistancePerCategory:
                        tmpCosineDistancePerCategory[key2] += value1 * value2[key1]
                    else:
                        tmpCosineDistancePerCategory[key2] = 0
                
        fileTestCosineDistancePerCategory[key] = tmpCosineDistancePerCategory
        count = count + 1
#         print count
                
            # print "File " + key + " calculation finished."
#             if key == '0011326':
#                 for key, val in fileTestCosineDistancePerCategory['0011326'].iteritems():
#                     print key + "\t" + str(val)   
                    
#     for key, val in fileTestCosineDistancePerCategory['0009613'].iteritems():
#         print key + "\t" + str(val)   
    
    # Count correctly distributed result
    correctCount = 0
    
    #key : fileName, value : {category : cosine score}
    for key, value in fileTestCosineDistancePerCategory.iteritems():
        
        maxScore = 0.0
        maxCategory = ""
        
        #key1 : category, value1 : cosine score
        for key1, value1 in value.iteritems():
            if value1 > maxScore:
                maxScore = value1
                maxCategory = key1

        if key1 in fileTestBelongCategory[key]:
            correctCount += 1
            #print key + ":" + key1
        
    print "Correct Result: " + str(correctCount)
        
    print "\n" + str(time.time() - startTime)
    print "\nTF-IDF Cosine Similarity Algorithm\n"

# Define TF-IDF based Cosine Similarity algorithm in Detail    
def tfidfCosineSimilarityDetail():
    print "\nTF-IDF Cosine Similarity Algorithm\n"

    
    
# Define Decision Tree algorithm.
def decisionTree(training_dict, testing_dict, file_category, words_name, use_sklearn_lib=True):
    print "\nDecision Tree Algorithm\n"
    if use_sklearn_lib:
        decisionTree_sklearn(training_dict, testing_dict, file_category)
    else:
        decisionTree_own_version(training_dict, testing_dict, file_category, words_name)

def decisionTree_sklearn(trainning_dict, testing_dict, file_category, criterion='gini', max_depth=100, draw=False):
    print "\nUsing sklearn library.... \n"
    # print trainning_dict
    X_train = []
    Y_train = []
    for category, termFreq_per_category in trainning_dict.items():
            X_train.append(termFreq_per_category)
            Y_train.append(category)

    clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    clf = clf.fit(X_train, Y_train)

    # predict and compute correct rate
    num_correct = 0
    total = len(testing_dict)
    for file, termFreq_per_file in testing_dict.items():
        predict = clf.predict(termFreq_per_file)
        if predict in file_category.get(file):
            num_correct += 1
    print 'The number of correct prediction is :' + str(num_correct) + ' and the total number is ' + str(total)
    print 'The correctness is ' + str(float(num_correct)/total)


    if draw:
        dot_data = StringIO.StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("DecisionTree.pdf")

def dict_to_list(dict):
    list = []
    for k, v in dict.items():
        v.insert(0, k)
        list.append(v)
    return list

def decisionTree_own_version(trainning_dict, testing_dict, file_category, words_name, num_categories=115, num_features=12212, min_node_size=1, max_node_depth=100):
    # max_categories: 115, max_features: 12212
    trainning_list = np.array(dict_to_list(trainning_dict))
    trainning_list = trainning_list[:num_categories]
    trainning_list = trainning_list[:, 0:num_features + 1]
    # print list

    words_name = words_name[0:num_features]
    words_name.insert(0, "category")
    df = pd.DataFrame(trainning_list, columns=words_name)

    g = DT.ClassificationTree()
    DT.set_global_data(df)
    parameters = dict()
    parameters['min_node_size'] = min_node_size
    parameters['max_node_depth'] = max_node_depth
    parameters['response'] = 'category'
    parameters['metric_kind'] = 'Entropy'
    g.train(parameters=parameters)
    g.plot()

    # make prediction
    num_correct = 0
    total = len(testing_dict)
    words_name = words_name[1:]
    for file, term in testing_dict.items():
        datapoint = pd.DataFrame(np.array([term[:num_features]]), columns=words_name)
        predict = g.predict(datapoint)
        if predict in file_category.get(file):
            num_correct += 1
    print 'The number of correct prediction is :' + str(num_correct)
    print 'The correctness is ' + str(float(num_correct)/total)

# Define Naive Bayes algorithm
def naiveBayes(list):
    print "\nNaive Bayes Algorithm\n"

# Define Naive Bayes algorithm in detail
def naiveBayesDetail(list):
    print "\nNaive Bayes Algorithm\n"

# Execute TF-IDF based Cosine Similarity algorithm    
# tfidfCosineSimilarity(termFrequencyPerCategoryList)2

# if '2' or 'k' or 'four' in wholeVocabularyFromTrainingAndTestSetList:
#     print "here"
# Execute Decision Tree algorithm
decisionTree(normalizedFrequencyPerCategoryInTrainingSetDict, normalizedFrequencyPerTestFileDict, fileTestBelongCategory, wholeVocabularyFromTrainingAndTestSetList)

# Execute NaiveBayes algorithm
naiveBayes(termFrequencyPerCategoryList)
    

