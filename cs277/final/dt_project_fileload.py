from nltk.stem.porter import *
import time
import cPickle as pickle
import math
import numpy as np



# import cs277.DecisionTree.DT as DT
import DecisionTree.DT as DT

#
# Pre-Process Part
#

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


startTime = time.time()
dataSet = prefixPath.split('/')[2]
print "Data Set to be used:\t" + dataSet
outputFile = open('pre_processed_data_object_decision_tree_' + dataSet + "_" + str(fileTestFractionSize), 'rb')

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

assignedCategoryTest = {}

#key: filename, val: categories that this test file belongs
for key, val in fileTestBelongCategory.iteritems():
    for cat in val:
        try:
            assignedCategoryTest[cat].append(key)
        except KeyError:
            assignedCategoryTest[cat] = []
            assignedCategoryTest[cat].append(key)

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
# print
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
        DT.decisionTree_version2(training_list, testing_list, fileTestBelongCategory)
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


# clf = DT.create_decision_tree(frequencyInFilePerCategoryInTrainingSetList, max_depth=80)
clf = DT.create_decision_tree(frequencyInFilePerCategoryInTrainingSetList)
top_k_categories = DT.get_top_k_prediction_class(clf, frequencyInFilePerCategoryInTestSetList[0], k=1)
# print top_k_categories
# print len(frequencyInFilePerCategoryInTestSetList)

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
    num_correct = 0

    # strArr = ""

    for testList in frequencyInFilePerCategoryInTestSetList:
        fileName = testList[-1]

        # value: categories that this file belongs (Actual)
        for value in fileTestBelongCategory[fileName]:

            predictedCategory = clf.predict(testList[0:len(testList)-2])
            if predictedCategory in fileTestBelongCategory[fileName]:
                num_correct += 1

            # predictedCategoryList = DT.get_top_k_prediction_class(clf, testList, 1)
            # predictedCategory = predictedCategoryList[0]
            confusionMatrix[categoryTestToIndexDict[value],categoryTestToIndexDict[predictedCategory[0]]] += 1

    print "number of correct result:\t" + str(num_correct)
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

    #Accuracy : (TP) / TP + FN
    microAverageAccuracy = float(confusionTable[0,0]) / float(confusionTable[0,0] + confusionTable[0,1])

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
                accuracyTestWithoutZeroFileCategory[key] = float(TP) / float(TP + FN)
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

# Execute NaiveBayes algorithm
# naiveBayes(termFrequencyPerCategoryList)
    

