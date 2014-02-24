from nltk.stem.porter import *
import time
import cPickle as pickle
import math


import cs277.DecisionTree.DT as DT
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

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'r') as f:
        return pickle.load(f)

# Define Decision Tree algorithm.
def decisionTree(training_list, testing_list, words_name, use_version2=True):
    print "-----------------------------------------------------------------------------------------"
    print "\nDecision Tree Algorithm\n"
    if use_version2:
        adjust_depth_dict = {}
        for max_depth in range(10, 121, 10):
            DT.decisionTree_version2(training_list, testing_list, max_depth=max_depth, adjust_depth_dict=adjust_depth_dict)
        save_obj(adjust_depth_dict, 'adjust_depth')
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
decisionTree(frequencyInFilePerCategoryInTrainingSetList, frequencyInFilePerCategoryInTestSetList, wholeVocabularyFromTrainingAndTestSetList)

# Execute NaiveBayes algorithm
naiveBayes(termFrequencyPerCategoryList)
    

