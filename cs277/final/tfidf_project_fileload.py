import os
import re
from nltk.corpus import stopwords
from nltk.stem.porter import *
from operator import itemgetter
import collections
import numpy as np
import time
import cPickle as pickle
import copy
import math
import matplotlib.pyplot as plt
import numpy
#
# Pre-Process Part
#

startTime = time.time()

# Set fraction size and prefix path
# File Fraction size to Read. Set between 0.1 and 1
fileFractionSize = 1
fileTestFractionSize = 1
prefixPath = "./dataset/Reuters21578-Apte-115Cat/"
# prefixPath = "./dataset/20news-bydate/"
# prefixPath = "./dataset/ohsumed-first-20000-docs/"

# Parameter : #1 - fraction size. #2 - dataSet
# Example: python project_save.py 1 ./dataset/Reuters21578-Apte-115Cat/

# Set FractionSize and Data Set to Use
if len(sys.argv) >= 2:
    if float(sys.argv[1]) > 0 and float(sys.argv[1]) <= 1:
        fileTestFractionSize = sys.argv[1]

if len(sys.argv) == 3:
    prefixPath = sys.argv[2]

dataSet = prefixPath.split('/')[2]
print "Data Set to be used:\t" + dataSet
outputFile = open('pre_processed_data_object_tfidf_' + dataSet + "_" + str(fileTestFractionSize), 'rb')

# File Fraction size to Read. Set between 0.1 and 1
fileFractionSize = pickle.load(outputFile)
fileTestFractionSize = pickle.load(outputFile)

print "Fraction to be used:\t" + str(fileTestFractionSize)

# Define Regular Expression to pre-process strings. Only AlphaNumeric and whitespace will be kept.
strPattern = re.compile('[^a-zA-Z0-9 ]')

# A dictionary which keeps token and its frequency for each category. It will keep a Dictionary in a Dictionary.
# key - category, value-{'term':frequency}
# Example : {'acq' : {'hi':1,'compu':3,'move':1 ...}}
categoryAlphaNumericStrStemmedDict = pickle.load(outputFile)
categoryTestAlphaNumericStrStemmedDict = pickle.load(outputFile)

# A dictionary which keeps token and its frequency for each file in each category for TEST set
# {category : { file : {term : frequency ...}}}
# Example : {acq : { 000056 : {'hi' : 1 , 'compu' : 3 ...}}}
categoryTestAllFileAlphaNumericStrStemmedDict = pickle.load(outputFile)

# A dictionary which keeps token, its frequency, and category for each file. It is layered Dictionary structure.
# 1st layer Dict {A}: key - category, value-{'term':frequency}
# 2nd layer Dict {B}: key - filename, value-{A}
# Example : {'000056' : {'acq' : {'hi':1, 'compu:3, 'move':1 ...}}}
fileAlphaNumericStrStemmedDict = pickle.load(outputFile)
fileTestAlphaNumericStrStemmedDict = pickle.load(outputFile)

# A dictionary which keeps test filename, and its categories in Set
# {'000056' : ('acq', 'alum')}
fileBelongCategory = pickle.load(outputFile)
fileTestBelongCategory = pickle.load(outputFile)

assignedCategoryTest = {}

#key: filename, val: categories that this test file belongs
for key, val in fileTestBelongCategory.iteritems():
    for cat in val:
        try:
            assignedCategoryTest[cat].append(key)
        except KeyError:
            assignedCategoryTest[cat] = []
            assignedCategoryTest[cat].append(key)


# for key, val in fileTestBelongCategory.iteritems():
#     print key + "\t" + str(len(val)) + "\t" + str(val)
    
# print fileTestBelongCategory

# For entire vocabularies in the training set, create a dictionary that a list (value) which contains frequency per category (key)
# Orders of vocabularies are same for every list. The order is as same as that of in wholeVocabularyFromTrainingAndTestSetList.
# Example : { 'category' : '[frequency for 'said', frequency for 'mln' ...]', 'category' : '[frequency for 'said', frequency for 'mln' ...]'  
# normalizedFrequencyPerCategoryInTrainingSetDict = pickle.load(outputFile)

# frequencyInFilePerCategoryInTrainingSetList = pickle.load(outputFile)
# frequencyInFilePerCategoryInTestSetList = pickle.load(outputFile)

# For entire vocabularies in the test set, create a dictionary that a list (value) which contains frequency per file (key)
# Orders of vocabularies are same for every list. The order is as same as that of in wholeVocabularyFromTrainingAndTestSetList.
# Example : { '0001268' : '[frequency for 'said', frequency for 'mln' ...]', 'category' : '[frequency for 'said', frequency for 'mln' ...]'   

# normalizedFrequencyPerTestFileDict = pickle.load(outputFile)

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

# for key, val in wholeVocabularyFrequencyDict.iteritems():
#     print str(key) + "\t" + str(val)
# 
# print
# print
# 
# for key1, value1 in categoryAlphaNumericStrStemmedDict.iteritems():
#     for key, value in value1.iteritems():
#         print str(key1) + "\t" + str(key) + "\t" + str(value)

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



print "Object loading finished. Elapsed Time:\t" + str(round((time.time() - startTime),2))
print
# print categoryNum
# print fileNum
# print categoryTestNum
# print fileTestNum

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

# test file and cosine score between it and category
# {file : {cat1: 0.89, cat2 : 0.94}}
fileTestCosineDistancePerCategory = {}

# test file and cosine score between it and category. ordered by score (desc).
# {file : [cat1, cat2, ... ]}
fileTestCosineDistanceOrderedByScorePerCategory = {}

categoryTestCosineDistanceForFilesInIt = {}
categoryTestCosineDistanceForFilesInItWithoutScoreList = {}

# {file:it's predicted category}
fileTestPredictionResultDict = {}

# Define TF-IDF based Cosine Similarity algorithm
def tfidfCosineSimilarity(list):
    global categoryAlphaNumericStrStemmedDict, categoryAlphaNumericStrStemmedNormalizedTFDict, wholeVocabularyIDFDict
    totalFrequencyCountPerCategory = {}
    
    startT = time.time()
    
    # Calculating Normalized TF for each category
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

    # Calculating Inverse Document Frequency (IDF) for each term
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
    # key: category, value: {term:frequency}
    for key,value in categoryAlphaNumericStrStemmedNormalizedTFDict.iteritems():
        
        tmpTFIDFDictPerCategory = {}
        tmpTFIDFUnitVectorDictPerCategory = {}
        tmpTFIDFDistance = 0

        # key1: term, value1 : frequency
        for key1, value1 in value.iteritems():
            tmp = value1 * wholeVocabularyIDFDict[key1]
            tmpTFIDFDictPerCategory[key1] = tmp
            tmpTFIDFDistance += tmp * tmp
            
        categoryAlphaNumericStrStemmedTFIDFDict[key] = tmpTFIDFDictPerCategory
    
        # Create unit Vector
        # key1: term, value1 : frequency
        for key1, value1 in value.iteritems():
            tmpTFIDFUnitVectorDictPerCategory[key1] = tmpTFIDFDictPerCategory[key1] / math.sqrt(float(tmpTFIDFDistance))
            
        categoryAlphaNumericStrStemmedTFIDFUnitVectorDict[key] = tmpTFIDFUnitVectorDictPerCategory
   
#     for key1, value1 in categoryAlphaNumericStrStemmedTFIDFUnitVectorDict['livestock'].iteritems():
#         print key1 + "\t" + str(value1)

   
    print "[TF-IDF] Training Time:\t" + str(round((time.time() - startT),2))

    startTestT = time.time()
        
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
                if key2 in wholeVocabularyIDFDict:
                    tmp = tmp * wholeVocabularyIDFDict[key2]
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
                # Not Found
                else:
                    if not key2 in tmpCosineDistancePerCategory:
                        tmpCosineDistancePerCategory[key2] = 0
                
        fileTestCosineDistancePerCategory[key] = tmpCosineDistancePerCategory
        count = count + 1
#         print count
                
        # print "File " + key + " calculation finished."
        # if key == '0012670':
        #     print fileTestCosineDistancePerCategory[key]
                    
#     for key, val in fileTestCosineDistancePerCategory['0009613'].iteritems():
#         print key + "\t" + str(val)   
    
    # Count correctly distributed result
    correctCount = 0
    
    #key : fileName, value : {category : cosine score}
    for key, value in fileTestCosineDistancePerCategory.iteritems():
        
        maxScore = -1
        maxCategory = ""

        tmpSorted = sorted(value, key=value.get, reverse=True)
        fileTestCosineDistanceOrderedByScorePerCategory[key] = tmpSorted

        #key1 : category, value1 : cosine score
        for key1, value1 in value.iteritems():

            # Create results of files for each category: {cat : {file : cosine-score}}
            try:
                categoryTestCosineDistanceForFilesInIt[key1][key] = value1
            except KeyError:
                categoryTestCosineDistanceForFilesInIt[key1] = {key : value1}

            # print key1 + "\t" + str(value1)
            # if value1 > maxScore:
            #     maxScore = value1
            #     maxCategory = key1

        # print key + " " + str(fileTestCosineDistanceOrderedByScorePerCategory[key])
        # print key
        # print str(fileTestCosineDistanceOrderedByScorePerCategory[key][0]) + " " + key
        maxScore = fileTestCosineDistancePerCategory[key][fileTestCosineDistanceOrderedByScorePerCategory[key][0]]
        maxCategory = fileTestCosineDistanceOrderedByScorePerCategory[key][0]

        if maxCategory in fileTestBelongCategory[key]:
            correctCount += 1
            #print key + ":" + key1

            # print key + "\t" + maxCategory + "\t" + "O" + "\t" + str(fileTestBelongCategory[key])
        # else:
            # print key + "\t" + maxCategory + "\t" + "X" + "\t" + str(fileTestBelongCategory[key])

    # create cat: file relationship. without cosine score.
    for key, val in categoryTestCosineDistanceForFilesInIt.iteritems():
        categoryTestCosineDistanceForFilesInItWithoutScoreList[key] = sorted(val, key=val.get, reverse=True)

    # create file : category relationship
    # key : category, val : list of files
    for key,val in categoryTestCosineDistanceForFilesInItWithoutScoreList.iteritems():
        for file in val:
            fileTestPredictionResultDict[file] = key

    # print fileTestCosineDistancePerCategory

    # print categoryTestCosineDistanceForFilesInItWithoutScoreList['acq']
    # print categoryTestCosineDistanceForFilesInIt['acq']

    print "[TF-IDF] Testing Time:\t" + str(round((time.time() - startTestT),2))
    print "[TF-IDF] Overall Time:\t" + str(round((time.time() - startT),2))
    print "[TF-IDF] Number of Correct Result:\t" + str(correctCount)
    print "[TF-IDF] Number of Total Result:\t" + str(len(fileTestCosineDistancePerCategory))
    print "[TF-IDF] Percentage of Simple Correct Result:\t" + str(float(correctCount) / float(len(fileTestCosineDistancePerCategory)))

    # print categoryTestCosineDistanceForFilesInItWithoutScoreList['acq']


# Return TOP k files for given categories
def returnTOPKfilesInTheCategory(cat, k=1):
    arrayLength = len(categoryTestCosineDistanceForFilesInItWithoutScoreList[cat])
    if k > arrayLength:
        return categoryTestCosineDistanceForFilesInItWithoutScoreList[cat][0:arrayLength]
    else:
        return categoryTestCosineDistanceForFilesInItWithoutScoreList[cat][0:k]

# Return TOP k categories for given filename
def createTfIdf(filename, k=1):
    arrayLength = len(fileTestCosineDistanceOrderedByScorePerCategory[filename])
    if k > arrayLength:
        return fileTestCosineDistanceOrderedByScorePerCategory[filename][0:arrayLength]
    else:
        return fileTestCosineDistanceOrderedByScorePerCategory[filename][0:k]


# Confusion Matrix - row : true test category, column - true column category
realCategorySize = len(categoryAlphaNumericStrStemmedDict.keys())
confusionMatrix = np.zeros((realCategorySize, realCategorySize), dtype=np.int)
categoryTestToIndexDict = {}
idx = 0

for key in categoryAlphaNumericStrStemmedDict.keys():
    categoryTestToIndexDict[key] = idx
    idx += 1

confusionTable = np.zeros((2,2), dtype=np.int)
confusionTableWithoutZeroFileCategory = np.zeros((2,2), dtype=np.int)
confusionTableArray = []

fScorePerCategoryTest = {}


# Create Confusion Matrix. row: Actual. col: Predicted
def createConfusionMatrix():

    # strArr = ""

    # key: filename, value: categories that this file belongs (Actual)
    for key, value in fileTestBelongCategory.iteritems():

        # key1 : each category that this file belongs (Actual)
        for key1 in value:
            predictedCategoryList = createTfIdf(key, 1)
            predictedCategory = predictedCategoryList[0]
            confusionMatrix[categoryTestToIndexDict[key1],categoryTestToIndexDict[predictedCategory]] += 1

    # for i in range(0,realCategorySize):
    #     for j in range(0,realCategorySize):
    #         strArr += str(confusionMatrix[i][j]) + " "
    #     print strArr
    #     strArr = ""

    # Now, create Confusion Table for each category
    # key: category, val: index
    sumOfAllOccurrence = sum(sum(confusionMatrix))
    betaForFScore = 1.0
    precisionTest = {}
    recallTest = {}
    accuracyTest = {}
    for key, val in categoryTestToIndexDict.iteritems():
        TP = confusionMatrix[val,val]
        FP = sum(confusionMatrix[:, val]) - TP
        FN = sum(confusionMatrix[val, :]) - TP
        TN = sumOfAllOccurrence - TP - FP - FN
        # print key + " " + str(TP) + " " + str(FP) + " Sum Positive:" + str(TP+FP) + " / " + str(FN) + " " + str(TN)

        tmpConfusionTable = np.array([TP,FN,FP,TN]).reshape(2,2)
        confusionTableArray.append(tmpConfusionTable)
        # print tmpConfusionTable
        # print sum(sum(tmpConfusionTable))
        # print


        # Prepare confusion Table for micro Average
        confusionTable[0,0] += TP
        confusionTable[0,1] += FN
        confusionTable[1,0] += FP
        confusionTable[1,1] += TN

        # Calculate F-score and store it per each category
        # F-Score: (1 + beta^2)*TP / { (1 + beta^2) * TP + beta^2 * FN + FP }
        tmpNominator = float(2 * TP + FN + FP)
        tmpFScore = 0.0

        # if nominator = 0, then we can't apply F-Score
        if tmpNominator > 0.0:
            tmpFScore = float(2 * TP) / tmpNominator

        # Put F-Score
        fScorePerCategoryTest[key] = tmpFScore

        # Precision
        if TP + FP != 0:
            precisionTest[key] = float(TP) / float(TP + FP)
        else:
            precisionTest[key] = 0.0

        # Recall
        if TP + FN != 0:
            recallTest[key] = float(TP) / float(TP + FN)
        else:
            recallTest[key] = 0.0

        # Accuracy
        if TP + FN != 0:
            accuracyTest[key] = float(TP) / float(TP + FN)
        else:
            accuracyTest[key] = 0.0

    print (confusionTable[0,0], confusionTable[0,1], confusionTable[1,0], confusionTable[1,1])
    #Precision : TP / (TP + FP)
    microAveragePrecision = float(confusionTable[0,0]) / float((confusionTable[0,0] + confusionTable[1,0]))

    macroAveragePrecision = np.mean(precisionTest.values())

    #Recall : TP / (TP + FN)
    microAverageRecall = float(confusionTable[0,0]) / float((confusionTable[0,0] + confusionTable[0,1]))

    macroAverageRecall = np.mean(recallTest.values())

    #Accuracy : (TP) / TP + FN
    microAverageAccuracy = float(confusionTable[0,0]) / float((confusionTable[0,0] + confusionTable[0,1]))

    macroAverageAccuracy = np.mean(accuracyTest.values())

    #Micro and Macro F-Score
    microAverageFScore = float((1 + betaForFScore * betaForFScore) * confusionTable[0,0]) / float(((1 + betaForFScore * betaForFScore) * confusionTable[0,0] + betaForFScore * betaForFScore * confusionTable[0,1] + confusionTable[1,0]))

    macroAverageFScore = np.mean(fScorePerCategoryTest.values())


    # Now, excluding TEST categories where number of test file is zero.
    betaForFScore = 1.0
    fScorePerCategoryTestWithoutZeroFileCategory = {}
    precisionTestWithoutZeroFileCategory = {}
    recallTestWithoutZeroFileCategory = {}
    accuracyTestWithoutZeroFileCategory = {}

    for key, val in categoryTestToIndexDict.iteritems():
        if key in categoryTestAlphaNumericStrStemmedDict and len(categoryTestAlphaNumericStrStemmedDict[key]) > 0:
            TP = confusionMatrix[val,val]
            FP = sum(confusionMatrix[:, val]) - TP
            FN = sum(confusionMatrix[val, :]) - TP
            TN = sumOfAllOccurrence - TP - FP - FN
            # print str(TP) + " " + str(FP) + " " + str(FN) + " " + str(TN)

            #tmpConfusionTable = np.array([TP,FN,FP,TN]).reshape(2,2)
            #confusionTableArray.append(tmpConfusionTable)

            # Prepare confusion Table for micro Average
            confusionTableWithoutZeroFileCategory[0,0] += TP
            confusionTableWithoutZeroFileCategory[0,1] += FN
            confusionTableWithoutZeroFileCategory[1,0] += FP
            confusionTableWithoutZeroFileCategory[1,1] += TN

            # Calculate F-score and store it per each category
            # F-Score: (1 + beta^2)*TP / { (1 + beta^2) * TP + beta^2 * FN + FP }
            tmpNominator = float(((1 + betaForFScore * betaForFScore) * TP + betaForFScore * betaForFScore * FN + FP))
            tmpFScore = 0.0

            # if nominator = 0, then we can't apply F-Score
            if tmpNominator > 0.0:
                tmpFScore = float((1 + betaForFScore * betaForFScore) * TP) / tmpNominator

            # Put F-Score
            fScorePerCategoryTestWithoutZeroFileCategory[key] = tmpFScore

            # Precision
            if TP + FP != 0:
                precisionTestWithoutZeroFileCategory[key] = float(TP) / float((TP + FP))
            else:
                precisionTestWithoutZeroFileCategory[key] = 0.0

            # Recall
            if TP + FN != 0:
                recallTestWithoutZeroFileCategory[key] = float(TP) / float((TP + FN))
            else:
                recallTestWithoutZeroFileCategory[key] = 0.0

            # Accuracy
            if TP + FN != 0:
                accuracyTestWithoutZeroFileCategory[key] = float(TP) / float((TP + FN))
            else:
                accuracyTestWithoutZeroFileCategory[key] = 0.0

    #Precision : TP / (TP + FP)
    microAveragePrecisionWithoutZeroFileCategory = float(confusionTableWithoutZeroFileCategory[0,0]) / float((confusionTableWithoutZeroFileCategory[0,0] + confusionTableWithoutZeroFileCategory[1,0]))

    macroAveragePrecisionWithoutZeroFileCategory = np.mean(precisionTestWithoutZeroFileCategory.values())

    #Recall : TP / (TP + FN)
    microAverageRecallWithoutZeroFileCategory = float(confusionTableWithoutZeroFileCategory[0,0]) / float((confusionTableWithoutZeroFileCategory[0,0] + confusionTableWithoutZeroFileCategory[0,1]))

    macroAverageRecallWithoutZeroFileCategory = np.mean(recallTestWithoutZeroFileCategory.values())

    #Accuracy : (TP) / TP + FN
    microAverageAccuracyWithoutZeroFileCategory = float(confusionTableWithoutZeroFileCategory[0,0]) / float((confusionTableWithoutZeroFileCategory[0,0] + confusionTableWithoutZeroFileCategory[0,1]))

    macroAverageAccuracyWithoutZeroFileCategory = np.mean(accuracyTestWithoutZeroFileCategory.values())

    # F-Score
    microAverageFScoreWithoutZeroFileCategory = float((1 + betaForFScore * betaForFScore) * confusionTableWithoutZeroFileCategory[0,0]) / float(((1 + betaForFScore * betaForFScore) * confusionTableWithoutZeroFileCategory[0,0] + betaForFScore * betaForFScore * confusionTableWithoutZeroFileCategory[0,1] + confusionTableWithoutZeroFileCategory[1,0]))

    macroAverageFScoreWithoutZeroFileCategory = np.mean(fScorePerCategoryTestWithoutZeroFileCategory.values())

    print
    print "[Confusion Matrix] Micro Average Precision:\t" + str(microAveragePrecision)
    print "[Confusion Matrix] Micro Average Recall:\t" + str(microAverageRecall)
    print "[Confusion Matrix] Micro Average Accuracy:\t" + str(microAverageAccuracy)
    print "[Confusion Matrix] Micro Average F-Score:\t" + str(microAverageFScore)
    print
    print "[Confusion Matrix] Macro Average Precision:\t" + str(macroAveragePrecision)
    print "[Confusion Matrix] Macro Average Recall:\t" + str(macroAverageRecall)
    print "[Confusion Matrix] Macro Average Accuracy:\t" + str(macroAverageAccuracy)
    print "[Confusion Matrix] Macro Average F-Score:\t" + str(macroAverageFScore)
    print
    print "[Confusion Matrix] Micro Average Precision (0 file Category excluded):\t" + str(microAveragePrecisionWithoutZeroFileCategory)
    print "[Confusion Matrix] Micro Average Recall (0 file Category excluded):\t" + str(microAverageRecallWithoutZeroFileCategory)
    print "[Confusion Matrix] Micro Average Accuracy (0 file Category excluded):\t" + str(microAverageAccuracyWithoutZeroFileCategory)
    print "[Confusion Matrix] Micro Average F-Score (0 file Category excluded) :\t" + str(microAverageFScoreWithoutZeroFileCategory)
    print
    print "[Confusion Matrix] Macro Average Precision (0 file Category excluded):\t" + str(macroAveragePrecisionWithoutZeroFileCategory)
    print "[Confusion Matrix] Macro Average Recall (0 file Category excluded):\t" + str(macroAverageRecallWithoutZeroFileCategory)
    print "[Confusion Matrix] Macro Average Accuracy (0 file Category excluded):\t" + str(macroAverageAccuracyWithoutZeroFileCategory)
    print "[Confusion Matrix] Macro Average F-Score (0 file Category excluded) :\t" + str(macroAverageFScoreWithoutZeroFileCategory)

#     np.set_printoptions(threshold='nan')
# 
#     print "\nNow Printing Confusion Matrix...\n"
#     print repr(confusionMatrix)

    # print "[Confusion Matrix] Maximum F-Score (excluding Categories without Files):\t" + str(max(fScorePerCategoryTestWithoutZeroFileCategory.values()))
    # print "[Confusion Matrix] Minimum F-Score (excluding Categories without Files):\t" + str(min(fScorePerCategoryTestWithoutZeroFileCategory.values()))
    # print "[Confusion Matrix] Maximum F-Score:\t" + str(max(fScorePerCategoryTest.values()))
    # print "[Confusion Matrix] Minimum F-Score:\t" + str(min(fScorePerCategoryTest.values()))

    # print "[Confusion Matrix] Individual F-Score per Each Category:"
    # for key, val in fScorePerCategoryTest.iteritems():
    #     print str(key) + "\t" + str(val)

# Execute TF-IDF based Cosine Similarity algorithm
tfidfCosineSimilarity(termFrequencyPerCategoryList)

# Execute ConfusionMatrix
createConfusionMatrix()


# AUC PR curve
print "Test with AUCPR of 2 categories"
 
# test phase for AUC of tfidf
def create_CategoryAUC(categoryList):  
    # {'acq':['1', '2'], 'cad':['3', '4'] ...}
    categoryAssigFileTFIDF = {} 
   
    for cat in categoryList: 
        categoryAssigFileTFIDF[cat] = returnTOPKfilesInTheCategory(cat, len(fileTestAlphaNumericStrStemmedDict.keys()))
                
    return categoryAssigFileTFIDF      
  
# call function for test data set 
categoryTestList  = assignedCategoryTest.keys()
categoryAssigFileTFIDF = create_CategoryAUC(categoryTestList)   

# compute tp, fn, fp, tn, result: 'acq' : ['001', '002','31313'] based on assignedCategoryTest
def get_tpfn_fptn(predFiles, trueFiles): 
           
    tp = len(set(trueFiles).intersection(set(predFiles)))
    fn = len(set(trueFiles).difference(set(predFiles)))
    fp = len(set(predFiles).difference(set(trueFiles)))
    tn = len(fileTestAlphaNumericStrStemmedDict) - (tp + fn + fp)
    return tp, fn, fp, tn
        
        
# compute precision and recall
def get_precision_recall(_predFiles, _trueFiles):   
    tp, fn, fp, tn = get_tpfn_fptn(_predFiles, _trueFiles)
    pre = float(tp)/(tp + fp)
    recall = float(tp)/(tp + fn)
   
    return pre, recall      
   
   
# return list of precision and recall based on value of k    
def get_precisons_recalls(_category):
           
    trueFiles = assignedCategoryTest[_category] 
    precisions =[]
    recalls =[]
        
    for k in numpy.arange(1, len(fileTestAlphaNumericStrStemmedDict.keys()) + 1, 1):
        # get k documents
        predFiles = categoryAssigFileTFIDF[_category][0:k]    
        #print predFiles
        precision, recall= get_precision_recall(predFiles, trueFiles)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls
  
#check with each category
count = 0
for i in range(20):
#for category in [categoryTestList[0], categoryTestList[1], categoryTestList[19], categoryTestList[21]]:   
    category = categoryTestList[i]                
    count += 1
    precisions, recalls = get_precisons_recalls(category) 
    fig = plt.figure(count)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(category)
    plt.plot(recalls, precisions)
    fig.show()
plt.show()

print "DONE"




