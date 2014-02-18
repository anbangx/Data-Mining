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
trainingOutputFile = open('training_naiveBayes_parameters', 'wb')

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


#[Naive Bayes YEN]
assignedCategory = pickle.load(outputFile)
assignedCategoryTest = pickle.load(outputFile)
fileAssignedCategory = pickle.load(outputFile)
fileAssignedCategoryTest = pickle.load(outputFile)
categoryList = pickle.load(outputFile)
categoryTestList = pickle.load(outputFile)
wordFrequencyInFile = pickle.load(outputFile)
wordFrequencyInFileTest = pickle.load(outputFile)


# print str(time.time() - startTime)
# print
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



# Naive Bayes Training phase return p(c), p(w|c), V, _fractionOfFilesInEachCategoryDict = Nc/N
def trainNB(_fractionOfFilesInEachCategoryDict, _categoryAlphaNumericStrStemmedDict, _wholeVocabularyList):
    print "\n Naive Bayes traning phase: "
    # get probability of each category {'acq':0.04, 'copper':0.1,...}
    prob_category = _fractionOfFilesInEachCategoryDict
    
    # count words in category c and store in dictionary numWord_category = {'acq': 121313, 'copper' : 24242,...}
    # and get conditional probabilities of each word in Vocabulary
    numWord_category = {}  # Nc
    prob_conditional = {}; # p(w|c) = nw/Nc
    
    for category in _categoryAlphaNumericStrStemmedDict.keys():
        
        numWord_category[category] = 0;
        for word in _categoryAlphaNumericStrStemmedDict[category].keys():
            numWord_category[category] = numWord_category[category] + _categoryAlphaNumericStrStemmedDict[category][word];
          

        prob_word = {}
        for word in _wholeVocabularyList:
            if(word in _categoryAlphaNumericStrStemmedDict[category].keys()): 
                # alpha = 2
                prob_word[word] = float(_categoryAlphaNumericStrStemmedDict[category][word] + 100)/(numWord_category[category] + 100*len(_wholeVocabularyList))                 
            else:
                prob_word[word] = float(100)/(numWord_category[category] + 100*len(_wholeVocabularyList))                
                
#             if(word in _categoryAlphaNumericStrStemmedDict[category].keys()): 
#                 # alpha = 2
#                 prob_word[word] = float(_categoryAlphaNumericStrStemmedDict[category][word] + 1)/(numWord_category[category] + 1000)                 
#             else:
#                 prob_word[word] = float(1)/(numWord_category[category] + 1000)
                
        prob_conditional[category] = prob_word 
   
    # get vocabulary
    vocabulary = _wholeVocabularyList;
    return prob_category, prob_conditional, vocabulary
  
# call training to get parameters
prob_category, prob_conditional, vocabulary =   trainNB(fractionOfFilesInEachCategoryDict, categoryAlphaNumericStrStemmedDict, wholeVocabularyList)
#store in file 'training_naiveBayes_parameters'
pickle.dump(assignedCategory, trainingOutputFile, -1)
pickle.dump(assignedCategoryTest,trainingOutputFile, -1 )
pickle.dump(fileAssignedCategory, trainingOutputFile, -1)
pickle.dump(fileAssignedCategoryTest, trainingOutputFile, -1)
pickle.dump(categoryList, trainingOutputFile, -1)
pickle.dump(categoryTestList,trainingOutputFile, -1 )
pickle.dump(wordFrequencyInFile, trainingOutputFile, -1)
pickle.dump(wordFrequencyInFileTest,trainingOutputFile, -1 )
pickle.dump(prob_category,trainingOutputFile, -1 )
pickle.dump(prob_conditional,trainingOutputFile, -1 )
pickle.dump(vocabulary,trainingOutputFile, -1 ) 

endTime = time.time()
print "Time : " + str(endTime - startTime)
print "\n DONE!"

    

