#A LOT OF REFERENCES AND EXPLANATIONS CAN BE READ HERE: http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
#THIS IS SCRIPT IS THE WAY TO GENERATE A DOC2VEC MODEL WITH A TRAINING SET, AND THEN IS POSSIBLE TO INFER VECTORS
#FOR THE TEST SET (TRANSFORM TWEET TEST SETS IN VECTORS, USING THE TRAINING MODEL)

#TO USE PRETRAINING WORD EMBEDDING, IS FUNDAMENTAL TO USE GENSIM <= 3.2.0
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

def createTestingVectors(InputDatasetsTesting, modelTraining):
    tweetVector = []
    labelVector = []
    for key, singleDataset in InputDatasetsTesting.items():
        for line, label in zip(singleDataset["Text"], singleDataset["Informativeness"]):
            tweetLineTokenize = line.strip().split()
            tweetVector.append(modelTraining.infer_vector(tweetLineTokenize, alpha=0.01, steps=1000))
            intLabel = 0
            if label == 'Related and informative':
                intLabel = 1
            labelVector.append(intLabel)
    bothVectors = [tweetVector,labelVector]
    return bothVectors


class LabeledLineSentence(object):
    #object constructor
    def __init__(self,InputDatasets):
        self.datasets = InputDatasets

    def __iter__(self):
        InputDatasets = self.datasets
        #iteration over a dictionary in which every value is a DataFrame
        for key, singleDataset in InputDatasets.items():
            for item_no, line, label in zip(range(0, len(singleDataset.index)), singleDataset["Text"],singleDataset["Informativeness"]):
                #yield is like return but for generators (for more details see: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do)
                yield LabeledSentence(utils.to_unicode(line).split(), [key + '_%s' % item_no + '_' + label])

    def to_array(self):
        self.sentences = []
        InputDatasets = self.datasets
        for key, singleDataset in InputDatasets.items():
            for item_no, line, label in zip(range(0, len(singleDataset.index)), singleDataset["Text"], singleDataset["Informativeness"]):
                self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [key + '_%s' % item_no + '_' + label]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


if __name__ == '__main__':

     # to extract only file names
     dirPath = "/Users/Flavio/Desktop/Tesi/Code/Crisis Tweet Analysis/MetaDataFinal"
     onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
     # to extract all the datasets from the directory

     InputDatasetsTraining = {}
     InputDatasetsTesting = {}

     nOfTrain = 6
     countTrain = 0


    # InputDatasets for training and for testing

     for file in onlyfiles:
        key = file[:7]     #extract a substring
        if "Flags" in file and countTrain <= nOfTrain:
            #to use to extract the right infos from the model
            InputDatasetsTraining[key] = pd.read_csv(join(dirPath,file))
            print(countTrain)
            print(file)
            countTrain = countTrain + 1
        elif "Flags" in file:
            InputDatasetsTesting[key] = pd.read_csv(join(dirPath,file))
            print(file)

     # The idea is: build a model with doc2vec (a vocabulary table) to obtain for each tweet a vector.
     # Then we select some of these vectors generated previously (which represent tweets) to use them as training set
     # Then we use the other ones left, in order to use them as test set
     # Certainly, it is fundamental to have the right label (relevant, not relevant, that in LogisticRegression will be 1 and 0) for each vector (tweet)
     # For this reason it has been very important to make a "mark" (representing the label) when the vectors have been generated
     # N.B. being Doc2Vec is an unsupervisioned learning, the doc2vec model can be created using all the datasets (of course no for the following classifier)

     # object from the class above, that is important to split correctly every line as document
     # it will be used to build the model

     # define the number of files that would be used as training set (there are at least 10 datasets)

     sentencesTraining = LabeledLineSentence(InputDatasetsTraining)


     # model creation (size is the length of the vector to generate)
     modelTraining = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=8)
     #to build the vocabular table from the sentences extracted from our dataframes
     modelTraining.build_vocab(sentencesTraining.to_array())

     #this vectors have dimension 300, so size of training is set to 300
     modelTraining.intersect_word2vec_format('./Doc2VecModels/pretrained_word_embeddings.txt')

     #training of the model
     for epoch in range(10):
         modelTraining.train(sentencesTraining.sentences_perm(), total_examples=modelTraining.corpus_count, epochs=modelTraining.iter)
     modelTraining.save('./Doc2VecModels/modelTweetsWithPreTrain.d2v')

     modelTraining = Doc2Vec.load('./Doc2VecModels/modelTweetsWithPreTrain.d2v')

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
             for i, label in zip(range(0,len(currentDataset.index)), currentDataset["Informativeness"]):
                # to use as training al the tweets related to a crisis, remove the comment in the line below (and indentate properly)
                #if label != "Not related":
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
     # train the classifier with the training
     classifier.fit(training_set, training_label)
     #print(training_set)
     #print(testVectors[0])
     #print(testVectors[1])
     print(classifier.score(testVectors[0], testVectors[1]))
