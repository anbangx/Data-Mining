'''
Created on Feb 19, 2014

@author: YENHOANG
'''
import re
import numpy
import time
import cPickle as pickle
import math
#
# Pre-Process Part
#

inputFile = open('pre_processed_data_object', 'rb')


# File Fraction size to Read. Set between 0.1 and 1
fileFractionSize = pickle.load(inputFile)
fileTestFractionSize = pickle.load(inputFile)

# Define Regular Expression to pre-process strings. Only AlphaNumeric and whitespace will be kept.
strPattern = re.compile('[^a-zA-Z0-9 ]')

# A dictionary which keeps token and its frequency for each category. It will keep a Dictionary in a Dictionary.
# key - category, value-{'term':frequency}
# Example : {'acq' : {'hi':1,'compu':3,'move':1 ...}}
categoryAlphaNumericStrStemmedDict = pickle.load(inputFile)
categoryTestAlphaNumericStrStemmedDict = pickle.load(inputFile)

# A dictionary which keeps token, its frequency, and category for each file. It is layered Dictionary structure.
# 1st layer Dict {A}: key - category, value-{'term':frequency}
# 2nd layer Dict {B}: key - filename, value-{A}
# Example : {'000056' : {'acq' : {'hi':1, 'compu:3, 'move':1 ...}}}
fileAlphaNumericStrStemmedDict = pickle.load(inputFile)
fileTestAlphaNumericStrStemmedDict = pickle.load(inputFile)

# A list which keeps whole vocabularies throughout whole categories. It will be sorted.
# Example : ['current', 'curtail', 'custom', 'cut', 'cuurent', 'cvg', 'cwt', 'cypru', 'cyrpu', 'd', 'daili' ...]
wholeVocabularyList = pickle.load(inputFile)
wholeTestVocabularyList = pickle.load(inputFile)

wholeVocabularyFrequency = pickle.load(inputFile)
wholeTestVocabularyFrequency = pickle.load(inputFile)

# A dictionary which keeps entire vocabulary and its frequency across whole categories
# Example : {'current' : 110, 'said' : 10000 ...... }
wholeVocabularyFrequencyDict = pickle.load(inputFile)
wholeVocabularyTestFrequencyDict = pickle.load(inputFile)

# A dictionary which keeps number of files in each category
# Example : {'acq': 115, 'alum': 222 ...}
numberOfFilesInEachCategoryDict = pickle.load(inputFile)
numberOfFilesInEachCategoryTestDict = pickle.load(inputFile) 

# A dictionary which keeps fraction of [number of files in each category] / [number of entire files]
# Example : {'acq':0.015, 'alum':0.031 ...}
fractionOfFilesInEachCategoryDict = pickle.load(inputFile) 
fractionOfFilesInEachCategoryTestDict = pickle.load(inputFile) 


categoryNum = pickle.load(inputFile)
fileNum = pickle.load(inputFile)

categoryTestNum = pickle.load(inputFile)
fileTestNum = pickle.load(inputFile)


# A two dimensional List which keeps frequency of term per category. 
# row = category. column = frequency of each term in that category.
# For term list, we are using whole terms across entire categories.
# Example : category- acq, bop, term- 'commonplac', 'commonwealth', 'commun'
#           commonplac   commonwealth  commun
#    acq         7              2         0
#    bop         8              9         1 
termFrequencyPerCategoryList = pickle.load(inputFile)


#[Naive Bayes YEN]
assignedCategory = pickle.load(inputFile)
assignedCategoryTest = pickle.load(inputFile)
fileAssignedCategory = pickle.load(inputFile)
fileAssignedCategoryTest = pickle.load(inputFile)
categoryList = pickle.load(inputFile)
categoryTestList = pickle.load(inputFile)
wordFrequencyInFile = pickle.load(inputFile)
wordFrequencyInFileTest = pickle.load(inputFile)


 
# Naive Bayes Training phase return p(c), p(w|c), V, _fractionOfFilesInEachCategoryDict = Nc/N
def trainNB(_fractionOfFilesInEachCategoryDict, _categoryAlphaNumericStrStemmedDict, _wholeVocabularyList, alpha):
    # print "\n Naive Bayes traning phase: "
    # get probability of each category {'acq':0.04, 'copper':0.1,...}
    prob_category = _fractionOfFilesInEachCategoryDict
       
    # count words in category c and store in dictionary numWord_category = {'acq': 121313, 'copper' : 24242,...}
    # and get conditional probabilities of each word in Vocabulary
    numWord_category = {}  # Nc
    prob_conditional = {}; # p(w|c) = nw/Nc
       
    for category in _categoryAlphaNumericStrStemmedDict.keys():       
        numWord_category[category] = sum(categoryAlphaNumericStrStemmedDict[category].itervalues())
   
        prob_word = {}
        for word in _wholeVocabularyList:
            if(word in _categoryAlphaNumericStrStemmedDict[category].keys()): 
                # check with different value of alpha
                prob_word[word] = float(_categoryAlphaNumericStrStemmedDict[category][word] + alpha)/(numWord_category[category] + alpha*len(_wholeVocabularyList))                 
            else:
                prob_word[word] = float(alpha)/(numWord_category[category] + alpha*len(_wholeVocabularyList))             
        
        prob_conditional[category] = prob_word 
      
    # get vocabulary
    vocabulary = _wholeVocabularyList;
    return prob_category, prob_conditional, vocabulary
   
   
# function to compute precision
def evaluateResult(_fileAssignedCategoryNV, _fileAssignedCategory):
    count = 0;
    for fileName in _fileAssignedCategoryNV.keys():
        if _fileAssignedCategoryNV[fileName] in _fileAssignedCategory[fileName]:
            count = count + 1
    return count
   
# function to extract word from documents       
def extract(_vocabulary, _fileName, _wordFrequencyInFile):
    W = []
    for word in _wordFrequencyInFile[_fileName].keys():
        if word in _vocabulary:
            W.append(word)
    return W
    
    
#Naive Bayes test phase 
def testNB(_categoryList, _wordFrequencyInFile, _prob_category, _prob_conditional , _vocabulary):  
    #print "\n Naive Bayes test phase: "
    # {'acq': ['001', '002'],....}
    assignedCategoryNV = {} 
    # {'001': 'acq', '002': acq', ...}
    fileAssignedCategoryNV = {}  
    for category in _categoryList:
        assignedCategoryNV[category] =[]
    
    for testFile in _wordFrequencyInFile.keys():
        W = extract(_vocabulary, testFile, _wordFrequencyInFile)
        scoreFile = {}
        for category in _categoryList:
            scoreFile[category] = math.log10(_prob_category[category])
            for word in W:                 
                scoreFile[category] += math.log10(_prob_conditional[category][word])
        #score[testFile] =  scoreFile 
        finalCategory = max(scoreFile, key=scoreFile.get)
        assignedCategoryNV[finalCategory].append(testFile)   
        fileAssignedCategoryNV[testFile] = finalCategory      
            
    return assignedCategoryNV, fileAssignedCategoryNV
    
    
precisionTest =[]
precisionTraining =[]
timeTest = []
timeTrainning = [] 
   
# call training to get parameters
#for alpha in numpy.arange(0.21, 0.51, 0.01): 
for alpha in numpy.arange(0.52, 0.60, 0.01):
    print "loop alpha = " + str(alpha)
    prob_category, prob_conditional, vocabulary =   trainNB(fractionOfFilesInEachCategoryDict, categoryAlphaNumericStrStemmedDict, wholeVocabularyList, alpha)
    
    # test set     
    startTimeTest = time.time()   
    assignedCategoryTestNV, fileAssignedCategoryTestNV = testNB(categoryTestList, wordFrequencyInFileTest, prob_category, prob_conditional, vocabulary)
    # compute precision of Naive Bayes with test set
    percentTest = evaluateResult(fileAssignedCategoryTestNV, fileAssignedCategoryTest)/float(len(fileAssignedCategoryTest))
    #print " Precision of test "  + str(percentTest)
    endTimeTest = time.time()
     
    precisionTest.append(percentTest)    
    timeTest.append(endTimeTest - startTimeTest)
     
    # training set     
    startTimeTrain = time.time()
    assignedCategoryNV, fileAssignedCategoryNV = testNB(categoryList, wordFrequencyInFile, prob_category, prob_conditional, vocabulary) 
    # compute precision of Naive Bayes with training set  
    percentTraining = evaluateResult(fileAssignedCategoryNV, fileAssignedCategory)/float(len(fileAssignedCategory))
    endTimeTrain= time.time()
     
    precisionTraining.append(percentTraining)
    timeTrainning.append(endTimeTrain - startTimeTrain)  
   
  
print(numpy.arange(0.52, 0.60, 0.01))
print "\ntesting information"
print  precisionTest
print timeTest
print "\ntraining information"
print  precisionTraining
print timeTrainning

print "DONE ALL!"

