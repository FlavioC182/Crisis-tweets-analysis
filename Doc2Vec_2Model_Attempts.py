#A LOT OF REFERENCES AND EXPLANATIONS CAN BE READ HERE: http://linanqiu.github.io/2015/10/07/word2vec-sentiment/


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
                yield LabeledSentence(utils.to_unicode(line).split(), [key + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        InputDatasets = self.datasets
        for key, singleDataset in InputDatasets.items():
            for item_no, line, label in zip(range(0, len(singleDataset.index)), singleDataset["Text"], singleDataset["Informativeness"]):
                self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [key + '_%s' % item_no]))
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
     InputDatasetsTraining2 = {}

     nOfTrain = 7
     nOfTrain2 = 3
     countTrain = 0
     countTrain2 = 0

    # InputDatasets for training and for testing

     for file in onlyfiles:
        key = file[:7]     #extract a substring
        if "Flags" in file and countTrain < nOfTrain:
            #to use to extract the right infos from the model
            InputDatasetsTraining[key] = pd.read_csv(join(dirPath,file))
            countTrain = countTrain + 1
            print(countTrain)
            if countTrain2 < nOfTrain2:
                InputDatasetsTraining2[key] = pd.read_csv(join(dirPath,file))



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
     sentencesTraining2 = LabeledLineSentence(InputDatasetsTraining2)
     # model creation (size is the length of the vector to generate)
     modelTraining = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
     modelTraining2 = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
     #to build the vocabular table from the sentences extracted from our dataframes
     modelTraining.build_vocab(sentencesTraining.to_array())
     modelTraining2.build_vocab(sentencesTraining2.to_array())
     #training of the model
     for epoch in range(10):
         modelTraining.train(sentencesTraining, total_examples=modelTraining.corpus_count, epochs=modelTraining.iter)
         modelTraining2.train(sentencesTraining2, total_examples=modelTraining2.corpus_count, epochs=modelTraining2.iter)

     # some attempts: list of words similiar to this one


     print(modelTraining['2012_co_0'])
     print(modelTraining2['2012_co_0'])
