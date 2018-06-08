#A LOT OF REFERENCES AND EXPLANATIONS CAN BE READ HERE: http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
#THIS IS SCRIPT IS THE WAY TO GENERATE A DOC2VEC MODEL WITH A TRAINING SET, AND THEN IS POSSIBLE TO INFER VECTORS
#FOR THE TEST SET (TRANSFORM TWEET TEST SETS IN VECTORS, USING THE TRAINING MODEL)

# This script allows to build a Doc2Vec model starting from training dataset
# By means of this model, we traduce the test set in the vector form
# We use the training set to train the standard classifier (logistic regression)
# We test the classifier with the test set already converted in vectors


# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
# numpy
import numpy
# random
from random import shuffle
# classifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
pd.options.mode.chained_assignment = None  # default='warn'

#to suppress Future Warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# this function allow to convert the datasets selected as test set in a vectorial form, exploiting the generated model

def createTestingVectors(InputDatasetsTesting, modelTraining):
    tweetVector = []    # to save the tweets in their vector form
    labelVector = []    # to save their related label

    # The test set is store in a dictionary in which there are the selected dataset that we will exploit for testing purposes
    # key: the name (a substring) of the dataset file
    # value: the dataset stored in a DataFrame object
    for key, singleDataset in InputDatasetsTesting.items():
        # to iterate over the single dataframe
        # N.B. Informativeness is the the field where the label is stored
        for line, label in zip(singleDataset["Text"], singleDataset["Informativeness"]):
            # fill the testing set list
            # infer_vector allows to convert a specific list of words in a vector, exploiting the already built model
            tweetLineTokenize = line.strip().split()    # to obtain a list of words from the tweet
            tweetVector.append(modelTraining.infer_vector(tweetLineTokenize, alpha=0.01, steps=1000))
            # for logistic regression we need 2 integer labels
            intLabel = 0
            if label == 'Related and informative':
                # if a tweet is relevant has the value 1 as label
                intLabel = 1
            labelVector.append(intLabel)
    bothVectors = [tweetVector,labelVector]
    return bothVectors

# This class allow the construction of an object that will be used to build the Doc2Vec model
class LabeledLineSentence(object):
    #object constructor
    def __init__(self,InputDatasets):
        # to store the structure that contains the data (dataframes) that will be used to build the model
        self.datasets = InputDatasets

    # this object will be a generator not an iterable
    # this method is fundamental because during the training phase of the model, the object will be iterated
    def __iter__(self):
        InputDatasets = self.datasets
        # iteration over a dictionary in which every value is a DataFrame
        for key, singleDataset in InputDatasets.items():
            # iteration over the single dataframe
            for item_no, line, label in zip(range(0, len(singleDataset.index)), singleDataset["Text"],singleDataset["Informativeness"]):
                # LabeledSentence is the object used to build the model. In particular Doc2Vec needs to receive
                # the documents (tweets) as LabeledSentence items. A LabeledSentence object contains simply:
                # the document (which is always represented as a line) as a list of words (for this reason there will be a split)
                # a label used to identify this particular document (this is the document ID named in the theory, that is exploited in the model building)
                yield LabeledSentence(utils.to_unicode(line).split(), [key + '_%s' % item_no + '_' + label])
                # N.B. yield is like return but for generators (for more details see: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do)

    # this method is equal to the iter, but it returns the list of LabeledSentence (used to build the model)
    def to_array(self):
        self.sentences = []
        InputDatasets = self.datasets
        for key, singleDataset in InputDatasets.items():
            for item_no, line, label in zip(range(0, len(singleDataset.index)), singleDataset["Text"], singleDataset["Informativeness"]):
                self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [key + '_%s' % item_no + '_' + label]))
        return self.sentences

    # this method is used to shuffle the list of LabeledSentence in order to have a better training
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


if __name__ == '__main__':

     # to extract only file names
     dirPath = "/Users/Flavio/Desktop/Tesi/Code/Crisis Tweet Analysis/MetaDataFinal"
     # to extracts all the names of the directory specified
     onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
     # to extract all the datasets from the directory

     # used as dictionary because we will use the keys to generate the "label" required for LabeledSentence (which represents the document ID)
     InputDatasetsTraining = {}
     InputDatasetsTesting = {}

     # there are 10 datasets files in the directory. Here we can set the number of files we will use as
     # training set. The other ones will be used as test set
     nOfTrain = 6
     countTrain = 0


    # InputDatasets for training and for testing

     for file in onlyfiles:
        #extract a substring of the name to use as key of the dictionary (to use it to build the label in LabeledSentence)
        key = file[:7]
        # there are different versions of the same dataset in the directory. We select only Flags version
        #es: 2012_colorado_metadata_Flags.read_csv
        if "Flags" in file and countTrain <= nOfTrain:
            #to use to extract the right infos from the model
            InputDatasetsTraining[key] = pd.read_csv(join(dirPath,file))
            print(countTrain)
            print(file)
            countTrain = countTrain + 1
        elif "Flags" in file:
            InputDatasetsTesting[key] = pd.read_csv(join(dirPath,file))
            print(file)

     # The idea is: build a model with doc2vec to obtain for each tweet a vector.
     # In this script, all the tweets (documents) used to train the model will be used as training set for the classifier's training
     # Indeed it is important to remember that there will be 2 training: the first one for the doc2vec model, the second one
     # for the standard classifier (in order to do our sentiment analysis)
     # For this reason it has been very important to make a "mark" (representing the label) when the vectors have been generated
     # N.B. to build Doc2Vec model an unsupervisioned learning is performed

     # object from the class above, that is important to split correctly every line (document)
     # it will be used to build the model

     # define the number of files that would be used as training set (there are at least 10 datasets)

     # create the object
     sentencesTraining = LabeledLineSentence(InputDatasetsTraining)


     # model creation (size is the length of the vector to generate)
     modelTraining = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=8)

     # to build the vocabular table from the sentences (documents, in our case tweets) extracted from our dataframes
     # every word of a document and the document ID itself are codified in this vocabulary
     modelTraining.build_vocab(sentencesTraining.to_array())

     # to check the the similarity of a word:
     # it will be converted (looking at the vocabulary table) and it will be given as input of the neural network
     # the results will be all the words similar (so with same vector value) to the input
     #before training should be wrong
     print(modelTraining.most_similar('help'))

     # training of the model
     # to train the weights of the neural network
     # the every vector of weights will be a document
     for epoch in range(15):
         modelTraining.train(sentencesTraining.sentences_perm(), total_examples=modelTraining.corpus_count, epochs=modelTraining.iter)

     modelTraining.save('./Doc2VecModels/modelTweets3.d2v')

     modelTraining = Doc2Vec.load('./Doc2VecModels/modelTweets3.d2v')

     # this method exploit the doc2vec model to traduce the test tweets (tweets belonging to the datasets chosen as test) into vectors
     # more the model has been trained with lots of datas, more efficent will be this conversion
     # in addition, if there are words unknown from the model (that are not in the vocabulary) they will be ignored
     # and of course the conversion will be not so much precise

     testVectors = createTestingVectors(InputDatasetsTesting, modelTraining)
     #print(testVectors)

     # some attempts: list of words similiar to this one
     print(modelTraining.most_similar('help'))
     print()
     print(modelTraining.most_similar('good'))
     print()
     print(modelTraining.most_similar('sport'))
     print()
     #print(model['2012_co_0'])


     countTrain = 0
     # every element would be a vector (a transformed tweet)
     training_set = []
     training_label = []

     # the idea is to extract from the model built, a set of tweets (in the form of vector) to use as training set
     # we will take all the tweets of 'nOfTrain' datasets
     # of course, it is important to take the label too
     # label = 1 -> relevant
     # label = 0 -> not relevant
     for file in onlyfiles:
         key = file[:7]     #extract a substring
         if "Flags" in file and countTrain <= nOfTrain:
             #to use to extract the right infos from the model
             currentDataset = pd.read_csv(join(dirPath,file))
             # for each dataframe chosen as training set, we will take the corrispondent vector value of every tweet from the model
             for i, label in zip(range(0,len(currentDataset.index)), currentDataset["Informativeness"]):
                # to use as training all the tweets related to a crisis, remove the comment in the line below (and indentate properly)
                # if label != "Not related":
                prefix = key + '_%s' % i + '_' + label
                training_set.append(modelTraining[prefix])
                intLabel = 0
                if label == 'Related and informative':
                    intLabel = 1
                training_label.append(intLabel)
             print(countTrain)
             print(file)
             countTrain = countTrain + 1

     classifier = LogisticRegression()
     #classifier = svm.SVC()
     # train the classifier with the training
     classifier.fit(training_set, training_label)
     #print(training_set)
     #print(testVectors[0])
     #print(testVectors[1])
     print(classifier.score(testVectors[0], testVectors[1]))
