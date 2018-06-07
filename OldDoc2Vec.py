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

     #to extract only file names
     dirPath = "/Users/Flavio/Desktop/Tesi/Code/Crisis Tweet Analysis/MetaDataFinal"
     onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
     #to extract all the datasets from the directory
     InputDatasets = {}
     for file in onlyfiles:
         if "Flags" in file:
             key = file[:7]     #extract a substring
             InputDatasets[key] = pd.read_csv(join(dirPath,file))

     # The idea is: build a model with doc2vec (a vocabulary table) to obtain for each tweet a vector.
     # Then we select some of these vectors generated previously (which represent tweets) to use them as training set
     # Then we use the other ones left, in order to use them as test set
     # Certainly, it is fundamental to have the right label (relevant, not relevant, that in LogisticRegression will be 1 and 0) for each vector (tweet)
     # For this reason it has been very important to make a "mark" (representing the label) when the vectors have been generated
     # N.B. being Doc2Vec is an unsupervisioned learning, the doc2vec model can be created using all the datasets (of course no for the following classifier)

     # object from the class above, that is important to split correctly every line as document
     # it will be used to build the model
     sentences = LabeledLineSentence(InputDatasets)

     '''
     # model creation (size is the length of the vector to generate)
     model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
     #to build the vocabular table from the sentences extracted from our dataframes
     model.build_vocab(sentences.to_array())
     #training of the model
     for epoch in range(10):
         model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
     model.save('./modelTweets.d2v')
     '''
     model = Doc2Vec.load('./modelTweets.d2v')


     # some attempts: list of words similiar to this one
     print(model.most_similar('help'))
     print()
     print(model.most_similar('good'))
     print()
     print(model.most_similar('sport'))
     print()
     #print(model['2012_co_0'])

     # define the number of files that would be used as training set
     nOfTrain = 7
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
         if "Flags" in file and countTrain < nOfTrain:
             #to use to extract the right infos from the model
             currentDataset = pd.read_csv(join(dirPath,file))
             key = file[:7]     #extract a substring
             for i, label in zip(range(0,len(currentDataset.index)), currentDataset["Informativeness"]):
                # to use as training al the tweets related to a crisis, remove the comment in the line below (and indentate properly)
                #if label != "Not related":
                prefix = key + '_%s' % i + '_' + label
                training_set.append(model[prefix])
                intLabel = 0
                if label == 'Related and informative':
                    intLabel = 1
                training_label.append(intLabel)
                #a way to break the for loop when the number is reached
             countTrain = countTrain + 1
             print(countTrain)


     test_set = []
     test_label =[]

     nOfTest = 3
     countTest = 0

     #similarly as above, the remaining tweets will be used as test set
     for file in onlyfiles:
         if "Flags" in file:
             if countTest < nOfTest and nOfTrain <= 0:
                 #to use to extract the right infos from the model
                 currentDataset = pd.read_csv(join(dirPath,file))
                 key = file[:7]     #extract a substring
                 for i, label in zip(range(0,len(currentDataset.index)), currentDataset["Informativeness"]):
                     prefix = key + '_%s' % i + '_' + label
                     test_set.append(model[prefix])
                     intLabel = 0
                     if label == 'Related and informative':
                         intLabel = 1
                     test_label.append(intLabel)
                 #a way to break the for loop when the number is reached
                 countTest = countTest + 1
                 print(countTest)
             nOfTrain = nOfTrain - 1
             print(file)

     classifier = LogisticRegression()
     # train the classifier with the training
     classifier.fit(training_set, training_label)

     print(classifier.score(test_set, test_label))
