import os
import re
from nltk.corpus import stopwords
from nltk.stem.porter import *
import collections
import numpy
import time
import cPickle as pickle

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

# A List which keeps token and its frequency for each category. It will keep a dictionary in a list.
# Example : {[0] : 'acq', [1] : {'hi':1,'compu':3,'move':1 ...}}
categoryAlphaNumericStrStemmedList = pickle.load(outputFile)
categoryTestAlphaNumericStrStemmedList = pickle.load(outputFile)

# A List which keeps token and its frequency for each file. It will keep a dictionary in a list.
# Example : {[0] : 'acq', [1] : '000056', [2] : {'hi':1,'compu':3,'move':1 ...}}
fileAlphaNumericStrStemmedList = pickle.load(outputFile)
fileTestAlphaNumericStrStemmedList = pickle.load(outputFile)

# A list which keeps whole vocabularies throughout whole categories. It will be sorted.
# Example : ['current', 'curtail', 'custom', 'cut', 'cuurent', 'cvg', 'cwt', 'cypru', 'cyrpu', 'd', 'daili' ...]
wholeVocabularyList = pickle.load(outputFile)
wholeTestVocabularyList = pickle.load(outputFile)

wholeVocabularyFrequency = pickle.load(outputFile)
wholeTestVocabularyFrequency = pickle.load(outputFile)

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

wholeVocabularyFrequencyDict = {}

for category in categoryAlphaNumericStrStemmedList:
    tmpDict = category[1]
    for key, value in tmpDict.iteritems():
        tmp = wholeVocabularyFrequencyDict.get(key)
        if tmp == None:
            wholeVocabularyFrequencyDict[key] = value
        else:
            wholeVocabularyFrequencyDict[key] = tmp + value

wholeVocabularyTestFrequencyDict = {}

for category in categoryTestAlphaNumericStrStemmedList:
    tmpDict = category[1]
    for key, value in tmpDict.iteritems():
        tmp = wholeVocabularyTestFrequencyDict.get(key)
        if tmp == None:
            wholeVocabularyTestFrequencyDict[key] = value
        else:
            wholeVocabularyTestFrequencyDict[key] = tmp + value

print wholeVocabularyTestFrequencyDict

print len(wholeVocabularyList)
print len(wholeTestVocabularyList)

print wholeVocabularyFrequency
print wholeTestVocabularyFrequency

# Define TF-IDF based Cosine Similarity algorithm    
def tfidfCosineSimilarity(list):
    print "\nTF-IDF Cosine Similarity Algorithm\n"

# Define TF-IDF based Cosine Similarity algorithm in Detail    
def tfidfCosineSimilarityDetail(list):
    print "\nTF-IDF Cosine Similarity Algorithm\n"

# Define Decision Tree algorithm. 
def decisionTree(list):
    print "\nDecision Tree Algorithm\n"
    
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
decisionTree(termFrequencyPerCategoryList)

# Execute NaiveBayes algorithm
naiveBayes(termFrequencyPerCategoryList)
    

