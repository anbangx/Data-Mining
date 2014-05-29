'''
Created on Mar 7, 2014

@author: YENHOANG
'''
from operator import itemgetter
import re
import time
import cPickle as pickle
import copy
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

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
inputFile = open('pre_processed_data_object_naivebayes_' + dataSet + "_" + str(fileTestFractionSize), 'rb')

# File Fraction size to Read. Set between 0.1 and 1
fileFractionSize = pickle.load(inputFile)
fileTestFractionSize = pickle.load(inputFile)

print "Fraction to be used:\t" + str(fileTestFractionSize)

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

# A dictionary which keeps test filename, and its categories in Set
# {'000056' : ('acq', 'alum')}
fileBelongCategory = pickle.load(inputFile)
fileTestBelongCategory = pickle.load(inputFile)

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

print "Object loading finished. Elapsed Time:\t" + str(round((time.time() - startTime),2))
print
 
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
   
# function to extract word from documents       
def extract(_vocabulary, _fileName, _wordFrequencyInFile):
    W = []
    for word in _wordFrequencyInFile[_fileName].keys():
        if word in _vocabulary:
            W.append(word)
    return W

# function to compute precision
def evaluateResult(_fileAssignedCategoryNV, _fileAssignedCategory):
    count = 0;
    for fileName in _fileAssignedCategoryNV.keys():
        if _fileAssignedCategoryNV[fileName] in _fileAssignedCategory[fileName]:
            count = count + 1
    return count
 

          
# CALL TRAINING PHASE
print "Start training "
startTime = time.time() 
prob_category, prob_conditional, vocabulary =   trainNB(fractionOfFilesInEachCategoryDict, categoryAlphaNumericStrStemmedDict, wholeVocabularyList, 0.08)
endTime = time.time()
print " Execution Time (" + str(endTime - startTime) + " sec ) -  Training the Data Set"
    
    
#Naive Bayes test phase return list of category
def testNBMultiLabel(_categoryList, _wordFrequencyInFile, _prob_category, _prob_conditional , _vocabulary, _testFile, k):  
    # list of top k categories that class can belongs to [ 'a', 'b', 'c']
    W = extract(_vocabulary, _testFile, _wordFrequencyInFile)
    scoreCatergories = {}
    for category in _categoryList:
            scoreCatergories[category] = math.log10(_prob_category[category])
            for word in W:                 
                scoreCatergories[category] += math.log10(_prob_conditional[category][word])
    # convert to list of tuples
    temp = sorted(scoreCatergories.items(), key=lambda x: (-x[1], x[0]))
    # return k categories    
    result = [score[0] for score in temp[0:k]]
    return result   
   
   
def createNB(testFile, k):
    return testNBMultiLabel(categoryTestList, wordFrequencyInFileTest, prob_category, prob_conditional , vocabulary, testFile, k)

#Naive Bayes test phase 

def testNB(_wordFrequencyInFile, k):  
    # {'001': 'acq', '002': acq', ...}
    fileAssignedCategoryNV = {} 

    for testFile in _wordFrequencyInFile.keys(): 
        fileAssignedCategoryNV[testFile] = createNB(testFile, k)
             
    return fileAssignedCategoryNV    


# Compute ACCURACY each (sigma tpc)/N, with N = total documents in test set # each document belong to 1 class
fileAssignedCategoryNV = testNB(wordFrequencyInFileTest, 1)
# convert to categoryAssignedFileNV
def convertResult(_fileAssignedCategoryNV, _categoryTestList, _wordFrequencyInFileTest):
    categoryAssignedFileNV = {}
    for category in categoryTestList:    
        categoryAssignedFileNV[category] = []
    # update
    for testFile in wordFrequencyInFileTest.keys(): 
        categoryAssignedFileNV[fileAssignedCategoryNV[testFile][0]].append(testFile)
    return categoryAssignedFileNV

# compute tp for each category
def compute_tp(_fileAssignedCategoryNV, _categoryTestList, _wordFrequencyInFileTest, _assignedCategoryTest):
    tp = 0
    # firstly convert
    categoryAssignedFileNV = convertResult(_fileAssignedCategoryNV, _categoryTestList, _wordFrequencyInFileTest) 
    # then compute tp = _assignedCategoryTest intersection with categoryAssignedFileNV for each category
    for category in categoryAssignedFileNV.keys():
        tp = tp + len(set(categoryAssignedFileNV[category]).intersection(set(_assignedCategoryTest[category])))
    return tp

# tp = compute_tp(fileAssignedCategoryNV, categoryTestList, wordFrequencyInFileTest, assignedCategoryTest)
# print "Accuracy of Naive Bayes: "
# print float(tp)/len(wordFrequencyInFileTest.keys())




# check with 1 file
# print "return top 5 categories for file 0010415 "
# result = createNB('0010415', 5)
# print result

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

    startCMTime = time.time()

    # strArr = ""

    # key: filename, value: categories that this file belongs (Actual)
    for key, value in fileTestBelongCategory.iteritems():

        # key1 : each category that this file belongs (Actual)
        for key1 in value:
            predictedCategoryList = createNB(key, 1)
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

    # print (confusionTable[0,0], confusionTable[0,1], confusionTable[1,0], confusionTable[1,1])
    # Precision : TP / (TP + FP)
    microAveragePrecision = float(confusionTable[0,0]) / float((confusionTable[0,0] + confusionTable[1,0]))

    macroAveragePrecision = np.mean(precisionTest.values())

    #Recall : TP / (TP + FN)
    microAverageRecall = float(confusionTable[0,0]) / float((confusionTable[0,0] + confusionTable[0,1]))

    macroAverageRecall = np.mean(recallTest.values())

    #Accuracy : (TP + TN) / TP + TN + FP + FN
    microAverageAccuracy = float(confusionTable[0,0])/float(confusionTable[0,0] + confusionTable[0, 1])

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

    # print "[Confusion Matrix] Maximum F-Score (excluding Categories without Files):\t" + str(max(fScorePerCategoryTestWithoutZeroFileCategory.values()))
    # print "[Confusion Matrix] Minimum F-Score (excluding Categories without Files):\t" + str(min(fScorePerCategoryTestWithoutZeroFileCategory.values()))
    # print "[Confusion Matrix] Maximum F-Score:\t" + str(max(fScorePerCategoryTest.values()))
    # print "[Confusion Matrix] Minimum F-Score:\t" + str(min(fScorePerCategoryTest.values()))

    # print "[Confusion Matrix] Individual F-Score per Each Category:"
    # for key, val in fScorePerCategoryTest.iteritems():
    #     print str(key) + "\t" + str(val)

    print "Time for Testing:\t" + str(round(time.time()-startCMTime,3))

    np.set_printoptions(threshold='nan')

    print "\nNow Printing Confusion Matrix...\n"
    print repr(confusionMatrix)

# Execute ConfusionMatrix
createConfusionMatrix()


# FOR AUC PR CURVE, IF YOU WANT TO CHECK IT YOU CAN UNCOMMENT FOLLOWS PARAGRAPH
'''
print "Test with AUCPR of 4 categories"
def extractWordAllFiles(_vocabulary, _wordFrequencyInFile):
    W = {}
    for _testFile in _wordFrequencyInFile.keys():
        W[_testFile] = extract(_vocabulary, _testFile, _wordFrequencyInFile)
    return W 
  
  
# We check files in each category
def testNBMultiLabel_CategoryAUC(_wordFrequencyInFile, _prob_conditional ,_vocabulary,_word,_category):
    # NEED TO REDUCE EXTRACT W    
    scoreFiles = {}
    for _testFile in _wordFrequencyInFile.keys():
            scoreFiles[_testFile] = 0
            for word in _word[_testFile]:                 
                scoreFiles[_testFile] += math.log10(_prob_conditional[_category][word])              
                  
    # convert to list of tuples
    temp = sorted(scoreFiles.items(), key=lambda x: (-x[1], x[0]))
    # return k categories    
    result = [score[0] for score in temp[0: len(_wordFrequencyInFile.keys())]]
          
    return result  
  
# extract all words in test files
word =  extractWordAllFiles(vocabulary, wordFrequencyInFileTest)
  
def createNB_CategoryAUC(_category):
    return testNBMultiLabel_CategoryAUC(wordFrequencyInFileTest, prob_conditional , vocabulary, word, _category)
   
#Naive Bayes test phase for AUC
def testNB_CategoryAUC(_categoryList):  
    # {'acq':{'1', '2'}, {'cad': '3', '4'} ...}
    categoryAssigFileNV = {} 
   
    for _category in _categoryList: 
        categoryAssigFileNV[_category] = createNB_CategoryAUC(_category)
                
    return categoryAssigFileNV      
  
# call function for test data set
categoryAssigFileNV = testNB_CategoryAUC(categoryTestList)    
  
# test dataset with nonEmpty Category
nonEmptyCatTest = []
for category in categoryTestList:
    if (len(assignedCategoryTest[category])>0):
        nonEmptyCatTest.append(category)
  
# compute tp, fn, fp, tn, result: 'acq' : ['001', '002','31313'] based on assignedCategoryTest
def get_tpfn_fptn(predFiles, trueFiles): 
           
    tp = len(set(trueFiles).intersection(set(predFiles)))
    fn = len(set(trueFiles).difference(set(predFiles)))
    fp = len(set(predFiles).difference(set(trueFiles)))
    tn = len(wordFrequencyInFileTest) - (tp + fn + fp)
    return tp, fn, fp, tn
        
        
# compute precision and recall
def get_precision_recall(_predFiles, _trueFiles):   
    tp, fn, fp, tn = get_tpfn_fptn(_predFiles, _trueFiles)
    pre = float(tp)/(tp + fp)
    recall = float(tp)/(tp + fn)
    accuracy = float((tp + tn))/ (tp + fn + fp + tn)
    return pre, recall, accuracy      
   
   
# return list of precision and recall based on value of k    
def get_precisons_recalls(_category):
       
    trueFiles = assignedCategoryTest[_category] 
    precisions =[]
    recalls =[]
        
    for k in numpy.arange(1, len(wordFrequencyInFileTest.keys()) + 1, 1):
        # get k documents
        predFiles = categoryAssigFileNV[_category][0:k]    
        #print predFiles
        precision, recall, accuracy = get_precision_recall(predFiles, trueFiles)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls
  
#check with each category
count = 0
for category in [nonEmptyCatTest[0], nonEmptyCatTest[2], nonEmptyCatTest[19], nonEmptyCatTest[21]]:                   
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
'''