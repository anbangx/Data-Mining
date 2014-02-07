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
print "Starting...\n"

# Identify Category
categoryList =  os.listdir("./dataset/Reuters21578-Apte-115Cat/training/")
categoryTestList = os.listdir("./dataset/Reuters21578-Apte-115Cat/test/")

# StopWord Definition
stopwordsList = stopwords.words('english')
stemmer = PorterStemmer()
fileNum = 0
fileTestNum = 0
categoryNum = 0
categoryTestNum = 0
outputFile = open('pre_processed_data_object', 'wb')

# File Fraction size to Read. Set between 0.1 and 1
fileFractionSize = 1
fileTestFractionSize = 1

# Define Regular Expression to pre-process strings. Only AlphaNumeric and whitespace will be kept.
strPattern = re.compile('[^a-zA-Z0-9 ]')

# A List which keeps token and its frequency for each category. It will keep a dictionary in a list.
# Example : {[0] : 'acq', [1] : {'hi':1,'compu':3,'move':1 ...}}
categoryAlphaNumericStrStemmedList = []
categoryTestAlphaNumericStrStemmedList = []

# A List which keeps token and its frequency for each file. It will keep a dictionary in a list.
# Example : {[0] : 'acq', [1] : '000056', [2] : {'hi':1,'compu':3,'move':1 ...}}
fileAlphaNumericStrStemmedList = []
fileTestAlphaNumericStrStemmedList = []

# A list which keeps whole vocabularies throughout whole categories. It will be sorted.
# Example : ['current', 'curtail', 'custom', 'cut', 'cuurent', 'cvg', 'cwt', 'cypru', 'cyrpu', 'd', 'daili' ...]
wholeVocabularySet = set()
wholeVocabularyList = []
wholeTestVocabularySet = set()
wholeTestVocabularyList = []

wholeVocabularyFrequency = 0
wholeTestVocabularyFrequency = 0


# Read Training Data Set
print "\nReading Training data Set"
print "Elap(s)\Dur(s)\tCategory#\tName\t#ofFile\t#ofUniqueTerms\t#Frequency"

#Iterate each category and create vector space for each category
for category in categoryList:

    tmpTime = time.time()

    # Temporary code to reduce time to process. Eliminate when processing entire set
    
#     if category == 'acq' or category == '.DS_Store':
#         continue
#     if categoryNum == 10:
#         break
    
    fileInCategoryList = os.listdir("./dataset/Reuters21578-Apte-115Cat/training/" + category + "/")
    tmpCategoryAlphaNumericStrStemmedDictoinary = {}
    # categoryAlphaNumericStrStemmedList[categoryNum][0] = category
    categoryTmpColumn = []
    categoryTmpColumn.append(str(category))
    tmpFileNum = 0
    tmpFreqPerCategory= 0
    tmpNumberOfUniqueTermPerCategory = 0
    tmpNumberOfTermPerCategory = 0
    
    for fileToTrain in fileInCategoryList:
        fileToTrainPath = './dataset/Reuters21578-Apte-115Cat/training/' + category + '/' + fileToTrain
        
        # Check the file size and read some fraction of the file defined in "fileFractionSize" variable
        filesize = os.path.getsize(fileToTrainPath)
        chunkReadSize = int(round(filesize * fileFractionSize))
        f = open(fileToTrainPath)
        fileStr = f.read(chunkReadSize)
        fileTmpColumn = []
        fileTmpColumn.append(str(category))
        fileTmpColumn.append(str(fileToTrain))

        # Remove non alphanumeric characters in the chunk
        fileAlphaNumericStr = re.sub(strPattern, ' ', fileStr)

        # Convert to lower case
        fileAlphaNumericStr = fileAlphaNumericStr.lower()

        # Remove Stop Words and Tokenize the chunk into a List by using whitespace
        fileAlphaNumericStrNoStopWords = ' '.join([word for word in fileAlphaNumericStr.split() if word not in stopwordsList])
        fileAlphaNumericStrList = fileAlphaNumericStrNoStopWords.split()
#         fileAlphaNumericStrList = fileAlphaNumericStr.split()

        # Apply Porter Stemmer and Put token and frequency to One dictionary
        tmpFileAlphaNumericStrStemmedDictionary = {}

        # Create vector space (dictionary) for each category
        for words in fileAlphaNumericStrList:
            tmp = stemmer.stem(words)
            if tmpFileAlphaNumericStrStemmedDictionary.get(tmp) == None:
                tmpFileAlphaNumericStrStemmedDictionary[tmp] = 1
            else:
                tmpFileAlphaNumericStrStemmedDictionary[tmp] += 1
            if tmpCategoryAlphaNumericStrStemmedDictoinary.get(tmp) == None:
                tmpCategoryAlphaNumericStrStemmedDictoinary[tmp] = 1
            else:
                tmpCategoryAlphaNumericStrStemmedDictoinary[tmp] += 1
            tmpFreqPerCategory += 1    
            if tmp not in wholeVocabularySet:
                wholeVocabularySet.add(tmp)

        fileTmpColumn.append(tmpFileAlphaNumericStrStemmedDictionary)
        fileAlphaNumericStrStemmedList.append(fileTmpColumn)
        fileNum += 1
        tmpFileNum += 1
        
    categoryTmpColumn.append(tmpCategoryAlphaNumericStrStemmedDictoinary)
    categoryAlphaNumericStrStemmedList.append(categoryTmpColumn)
    categoryNum += 1
    wholeVocabularyFrequency += tmpFreqPerCategory
    
    print "%6.3g"%(time.time() - startTime) + "\t" + "%6.3g"%(time.time() - tmpTime) + "\t" + str(categoryNum) +  "\t" + category + "\t" + str(tmpFileNum) + "\t" + str(len(tmpCategoryAlphaNumericStrStemmedDictoinary)) + "\t" + str(tmpFreqPerCategory)


print "\nReading Test data Set"
print "Elap(s)\Dur(s)\tCategory#\tName\t#ofFile\t#ofUniqueTerms\t#Frequency"

#Iterate each TEST category and create vector space for each category
for categoryTest in categoryTestList:

    tmpTime = time.time()

    # Temporary code to reduce time to process. Eliminate when processing entire set
#     if categoryTest == 'acq' or categoryTest == '.DS_Store':
#         continue
#     if categoryTestNum == 10:
#         break
    
    fileInCategoryTestList = os.listdir("./dataset/Reuters21578-Apte-115Cat/test/" + categoryTest + "/")
    tmpCategoryTestAlphaNumericStrStemmedDictoinary = {}
    # categoryAlphaNumericStrStemmedList[categoryNum][0] = category
    categoryTestTmpColumn = []
    categoryTestTmpColumn.append(str(categoryTest))
    tmpFileTestNum = 0
    tmpFreqPerCategoryTest= 0
    tmpNumberOfUniqueTermPerCategoryTest = 0
    tmpNumberOfTermPerCategoryTest = 0
    
    for fileToTest in fileInCategoryTestList:
        fileToTestPath = './dataset/Reuters21578-Apte-115Cat/test/' + categoryTest + '/' + fileToTest
        
        # Check the file size and read some fraction of the file defined in "fileFractionSize" variable
        filesizeTest = os.path.getsize(fileToTestPath)
        chunkTestReadSize = int(round(filesizeTest * fileTestFractionSize))
        f = open(fileToTestPath)
        fileTestStr = f.read(chunkTestReadSize)
        fileTestTmpColumn = []
        fileTestTmpColumn.append(str(categoryTest))
        fileTestTmpColumn.append(str(fileToTest))

        # Remove non alphanumeric characters in the chunk
        fileTestAlphaNumericStr = re.sub(strPattern, ' ', fileTestStr)

        # Convert to lower case
        fileTestAlphaNumericStr = fileTestAlphaNumericStr.lower()

        # Remove Stop Words and Tokenize the chunk into a List by using whitespace
        fileTestAlphaNumericStrNoStopWords = ' '.join([word for word in fileTestAlphaNumericStr.split() if word not in stopwordsList])
        fileTestAlphaNumericStrList = fileTestAlphaNumericStrNoStopWords.split()

        # Apply Porter Stemmer and Put token and frequency to One dictionary
        tmpFileTestAlphaNumericStrStemmedDictionary = {}

        # Create vector space (dictionary) for each category
        for words in fileTestAlphaNumericStrList:
            tmp = stemmer.stem(words)
            if tmpFileTestAlphaNumericStrStemmedDictionary.get(tmp) == None:
                tmpFileTestAlphaNumericStrStemmedDictionary[tmp] = 1
            else:
                tmpFileTestAlphaNumericStrStemmedDictionary[tmp] += 1
            if tmpCategoryTestAlphaNumericStrStemmedDictoinary.get(tmp) == None:
                tmpCategoryTestAlphaNumericStrStemmedDictoinary[tmp] = 1
            else:
                tmpCategoryTestAlphaNumericStrStemmedDictoinary[tmp] += 1
            tmpFreqPerCategoryTest += 1    
            if tmp not in wholeTestVocabularySet:
                wholeTestVocabularySet.add(tmp)

        fileTestTmpColumn.append(tmpFileTestAlphaNumericStrStemmedDictionary)
        fileTestAlphaNumericStrStemmedList.append(fileTestTmpColumn)
        fileTestNum += 1
        tmpFileTestNum += 1
        
    categoryTestTmpColumn.append(tmpCategoryTestAlphaNumericStrStemmedDictoinary)
    categoryTestAlphaNumericStrStemmedList.append(categoryTestTmpColumn)
    categoryTestNum += 1
    wholeTestVocabularyFrequency += tmpFreqPerCategoryTest
    
    print "%6.3g"%(time.time() - startTime) + "\t" + "%6.3g"%(time.time() - tmpTime) + "\t" + str(categoryTestNum) +  "\t" + categoryTest + "\t" + str(tmpFileTestNum) + "\t" + str(len(tmpCategoryTestAlphaNumericStrStemmedDictoinary)) + "\t" + str(tmpFreqPerCategoryTest)


# Sort entire Vocabulary
wholeVocabularyList = list(wholeVocabularySet)
wholeVocabularyList.sort()

wholeTestVocabularyList = list(wholeTestVocabularySet)
wholeTestVocabularyList.sort()

print
print "Statistics of Entire Training data Set"
print "# of Categories:\t" + str(categoryNum)
print "# of Files:\t" + str(fileNum)
print "# of Vocabularies:\t" + str(len(wholeVocabularyList))
print "# of Frequency:\t" + str(wholeVocabularyFrequency)


# print
# print wholeVocabularyList

# for i in range(0,categoryNum):
#    print str(categoryAlphaNumericStrStemmedList[i][0]) + " ::::::: " + str(categoryAlphaNumericStrStemmedList[i][1])


# A two dimensional List which keeps frequency of term per category. 
# row = category. column = frequency of each term in that category.
# For term list, we are using whole terms across entire categories.
# Example : category- acq, bop, term- 'commonplac', 'commonwealth', 'commun'
#           commonplac   commonwealth  commun
#    acq         7              2         0
#    bop         8              9         1 
termFrequencyPerCategoryList = []

# Creating A two dimensional List which keeps frequency of term per category
for categoryRow in categoryAlphaNumericStrStemmedList:
    tmpColumn = []
    category = categoryRow[0]
    tmpColumn.append(category)
    categoryTermFreq = categoryRow[1]
    for term in wholeVocabularyList:
        tmp = categoryTermFreq.get(term)
        if tmp == None:
            tmpColumn.append(0)
        else:
            tmpColumn.append(tmp)
    termFrequencyPerCategoryList.append(tmpColumn)

pickle.dump(fileFractionSize, outputFile, -1)
pickle.dump(fileTestFractionSize, outputFile, -1)
pickle.dump(categoryAlphaNumericStrStemmedList, outputFile, -1)
pickle.dump(categoryTestAlphaNumericStrStemmedList, outputFile, -1)
pickle.dump(fileAlphaNumericStrStemmedList, outputFile, -1)
pickle.dump(fileTestAlphaNumericStrStemmedList, outputFile, -1)
pickle.dump(wholeVocabularyList, outputFile, -1)
pickle.dump(wholeTestVocabularyList, outputFile, -1)
pickle.dump(wholeVocabularyFrequency, outputFile, -1)
pickle.dump(wholeTestVocabularyFrequency, outputFile, -1)
pickle.dump(categoryNum, outputFile, -1)
pickle.dump(fileNum, outputFile, -1)
pickle.dump(categoryTestNum, outputFile, -1)
pickle.dump(fileTestNum, outputFile, -1)
pickle.dump(termFrequencyPerCategoryList, outputFile, -1)

# print termFrequencyPerCategoryList

print 

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
    

