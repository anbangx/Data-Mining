from nltk.stem.porter import *
import time
import cPickle as pickle
import math


import DecisionTree.DT as DT
#
# Pre-Process Part
#

startTime = time.time()
outputFile = open('pre_processed_data_object_decision_tree', 'rb')

# File Fraction size to Read. Set between 0.1 and 1
fileFractionSize = pickle.load(outputFile)
fileTestFractionSize = pickle.load(outputFile)

# Define Regular Expression to pre-process strings. Only AlphaNumeric and whitespace will be kept.
strPattern = re.compile('[^a-zA-Z0-9 ]')

# A dictionary which keeps token and its frequency for each category. It will keep a Dictionary in a Dictionary.
# key - category, value-{'term':frequency}
# Example : {'acq' : {'hi':1,'compu':3,'move':1 ...}}
# categoryAlphaNumericStrStemmedDict = pickle.load(outputFile)
# categoryTestAlphaNumericStrStemmedDict = pickle.load(outputFile)

# A dictionary which keeps token, its frequency, and category for each file. It is layered Dictionary structure.
# 1st layer Dict {A}: key - category, value-{'term':frequency}
# 2nd layer Dict {B}: key - filename, value-{A}
# Example : {'000056' : {'acq' : {'hi':1, 'compu:3, 'move':1 ...}}}
# fileAlphaNumericStrStemmedDict = pickle.load(outputFile)
# fileTestAlphaNumericStrStemmedDict = pickle.load(outputFile)

# A dictionary which keeps test filename, and its categories in Set
# {'000056' : ('acq', 'alum')}
fileBelongCategory = pickle.load(outputFile)
fileTestBelongCategory = pickle.load(outputFile)

# print fileTestBelongCategory

# For entire vocabularies in the training set, create a dictionary that a list (value) which contains frequency per category (key)
# Orders of vocabularies are same for every list. The order is as same as that of in wholeVocabularyFromTrainingAndTestSetList.
# Example : { 'category' : '[frequency for 'said', frequency for 'mln' ...]', 'category' : '[frequency for 'said', frequency for 'mln' ...]'  
# normalizedFrequencyPerCategoryInTrainingSetDict = pickle.load(outputFile)

frequencyInFilePerCategoryInTrainingSetList = pickle.load(outputFile)
frequencyInFilePerCategoryInTestSetList = pickle.load(outputFile)

# For entire vocabularies in the test set, create a dictionary that a list (value) which contains frequency per file (key)
# Orders of vocabularies are same for every list. The order is as same as that of in wholeVocabularyFromTrainingAndTestSetList.
# Example : { '0001268' : '[frequency for 'said', frequency for 'mln' ...]', 'category' : '[frequency for 'said', frequency for 'mln' ...]'   

# normalizedFrequencyPerTestFileDict = pickle.load(outputFile)

# Entire Vocubulary List which include every terms from the training set and test set.
wholeVocabularyFromTrainingAndTestSetList = pickle.load(outputFile)

# A list which keeps whole vocabularies throughout whole categories. It will be sorted.
# Example : ['current', 'curtail', 'custom', 'cut', 'cuurent', 'cvg', 'cwt', 'cypru', 'cyrpu', 'd', 'daili' ...]
# wholeVocabularyList = pickle.load(outputFile)
# wholeTestVocabularyList = pickle.load(outputFile)
# 
# wholeVocabularyFrequency = pickle.load(outputFile)
# wholeTestVocabularyFrequency = pickle.load(outputFile)

# A dictionary which keeps entire vocabulary and its frequency across whole categories
# Example : {'current' : 110, 'said' : 10000 ...... }
# wholeVocabularyFrequencyDict = pickle.load(outputFile)
# wholeVocabularyTestFrequencyDict = pickle.load(outputFile)

# A dictionary which keeps number of files in each category
# Example : {'acq': 115, 'alum': 222 ...}
# numberOfFilesInEachCategoryDict = pickle.load(outputFile)
# numberOfFilesInEachCategoryTestDict = pickle.load(outputFile) 

# A dictionary which keeps fraction of [number of files in each category] / [number of entire files]
# Example : {'acq':0.015, 'alum':0.031 ...}
# fractionOfFilesInEachCategoryDict = pickle.load(outputFile) 
# fractionOfFilesInEachCategoryTestDict = pickle.load(outputFile) 


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
# termFrequencyPerCategoryList = pickle.load(outputFile)

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

# Define TF-IDF based Cosine Similarity algorithm in Detail    
def tfidfCosineSimilarityDetail():
    print "\nTF-IDF Cosine Similarity Algorithm\n"

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'r') as f:
        return pickle.load(f)

# Define Decision Tree algorithm.
def decisionTree(training_list, testing_list, fileTestBelongCategory, words_name, use_version2=True):
    print "-----------------------------------------------------------------------------------------"
    print "\nDecision Tree Algorithm\n"
    if use_version2:
        # adjust_depth_dict = {}
        # for max_depth in range(10, 121, 10):
        #     DT.decisionTree_version2(training_list, testing_list, max_depth=max_depth, adjust_depth_dict=adjust_depth_dict)
        # save_obj(adjust_depth_dict, 'adjust_depth')
        DT.decisionTree_version2(training_list, testing_list, fileTestBelongCategory, max_depth=10)
    else:
        DT.decisionTree_version1(training_list, testing_list, words_name, num_trainning_file=200, num_features=1000) # num_trainning_file=len(training_list), num_features=len(training_list[0]) - 1

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
# tfidfCosineSimilarity(termFrequencyPerCategoryList)

# Execute Decision Tree algorithm
# decisionTree(frequencyInFilePerCategoryInTrainingSetList, frequencyInFilePerCategoryInTestSetList, fileTestBelongCategory, wholeVocabularyFromTrainingAndTestSetList)
clf = DT.create_decision_tree(frequencyInFilePerCategoryInTrainingSetList, max_depth=50)
top_k_categories = DT.get_top_k_prediction_class(clf, frequencyInFilePerCategoryInTestSetList[0], k=20)
print top_k_categories

# Execute NaiveBayes algorithm
# naiveBayes(termFrequencyPerCategoryList)
    

