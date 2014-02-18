'''
Created on Feb 9, 2014

@author: YENHOANG
'''
import time
import cPickle as pickle
import math
#
# Pre-Process Part
#

startTime = time.time()
trainingOutputFile = open('training_naiveBayes_parameters', 'rb')
# resultOutputFile = open('naiveBayes_result', 'wb')

assignedCategory = pickle.load(trainingOutputFile)
assignedCategoryTest = pickle.load(trainingOutputFile)
fileAssignedCategory = pickle.load(trainingOutputFile)
fileAssignedCategoryTest = pickle.load(trainingOutputFile)
categoryList = pickle.load(trainingOutputFile)
categoryTestList = pickle.load(trainingOutputFile)
wordFrequencyInFile = pickle.load(trainingOutputFile)
wordFrequencyInFileTest = pickle.load(trainingOutputFile)
prob_category = pickle.load(trainingOutputFile)
prob_conditional = pickle.load(trainingOutputFile)
vocabulary = pickle.load(trainingOutputFile)


 

# function to compute precision
def evaluateResult(_fileAssignedCategory, _fileAssignedCategoryNV):
    count = 0;
    for fileName in _fileAssignedCategory:
        if _fileAssignedCategory[fileName] == _fileAssignedCategoryNV[fileName]:
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
    print "\n Naive Bayes test phase: "
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
        fileAssignedCategoryNV[testFile] =finalCategory      
         
    return assignedCategoryNV, fileAssignedCategoryNV
 
 
# test 2 data sets: traiing and test set
assignedCategoryTestNV, fileAssignedCategoryTestNV = testNB(categoryTestList, wordFrequencyInFileTest, prob_category, prob_conditional, vocabulary)
assignedCategoryNV, fileAssignedCategoryNV = testNB(categoryList, wordFrequencyInFile, prob_category, prob_conditional, vocabulary)

# compute precision of Naive Bayes with 2 data sets
percentTest = evaluateResult(fileAssignedCategoryTest, fileAssignedCategoryTestNV)/float(len(fileAssignedCategoryTest))
print " Precision of test "  + str(percentTest)

percentTraining = evaluateResult(fileAssignedCategory, fileAssignedCategoryNV)/float(len(fileAssignedCategory))
print "Precision of training "  + str(percentTraining)

# save information in to files
# pickle.dump(assignedCategoryTest, resultOutputFile, -1)
# #pickle.dump(assignedCategoryTestNV,resultOutputFile, -1)
# #pickle.dump(scoreTraining,resultOutputFile, -1)
endTime = time.time()
print "Execute time " + str(endTime - startTime)
print "\n DONE!"

    

