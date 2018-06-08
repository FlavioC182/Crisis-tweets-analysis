#A LOT OF REFERENCES AND EXPLANATIONS CAN BE READ HERE: https://github.com/jhlau/doc2vec#pre-trained-doc2vec-models
#TO USE ONLY WITH PYTHON 2.X AND WITH OLD GENSIM (0.X.X)

# This script allow to use a pre trained doc2vec model, that can be used to represent tweets by means of vectors with
# specific properties.

# gensim modules
import gensim.models as g
import logging
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

if __name__ == '__main__':

     # to extract only file names
     dirPath = "/Users/Flavio/Desktop/Tesi/Code/Crisis Tweet Analysis/MetaDataFinal"
     # to extracts all the names of the directory specified
     onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
     # to extract all the datasets from the directory

     # used as dictionary, we could have used a simple list, but dictionary results to be fundamental
     # in other similar scripts, so we prefer to keep the same data structure

     InputDatasetsTraining = {}
     InputDatasetsTesting = {}

     # there are 10 datasets files in the directory. Here we can set the number of files we will use as
     # training set. The other ones will be used as test set
     nOfTrain = 7
     countTrain = 0


    # InputDatasets for training and for testing

     for file in onlyfiles:
        #extract a substring of the name to use as key of the dictionary
        key = file[:7]
        # there are different versions of the same dataset in the directory. We select only Flags version
        #es: 2012_colorado_metadata_Flags.read_csv
        if "Flags" in file and countTrain < nOfTrain:
            #to use to extract the right infos from the model
            InputDatasetsTraining[key] = pd.read_csv(join(dirPath,file))
            #some prints for checks
            print(countTrain)
            print(file)
            countTrain = countTrain + 1
        elif "Flags" in file:
            InputDatasetsTesting[key] = pd.read_csv(join(dirPath,file))
            print(file)

     # The idea is: exploit a pre trained doc2vec model to obtain for each tweet a vector.
     # As we have done before, we had splitted the tweets in two subset: training and test set
     # We traduce tweets in vectors (or better, we infer vectors) using the pre trained model
     # Then, we use the training set (in the form of vectors) to train a standard classifier (we have used Logistic Regression)
     # Finally, we test the accuracy on the classifier using the testing set
     # N.B. being Doc2Vec is an unsupervisioned learning, the doc2vec model can be created using all the datasets (of course no for the following classifier)

     #enable logging
     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


     #inference hyper-parameters for the metod infer_vector
     start_alpha=0.01
     infer_epoch=1000
     #path from which take the model
     path = './Doc2VecModels/doc2vec.bin'
     #load model
     model = g.Doc2Vec.load(path)

     test_set = []
     test_label = []

     training_set = []
     training_label = []

     # iterate ove the dictionary of datasets selected as training ones
     for key, singleDataset in InputDatasetsTraining.items():
         # for each dataset
         for line, label in zip(singleDataset["Text"], singleDataset["Informativeness"]):
             # to split the tweet text in a list of words
             tweetLineTokenize = line.strip().split()
             # fill the training set vector
             # infer_vector allows to convert a specific list of words in a vector, exploiting the already built model
             training_set.append(model.infer_vector(tweetLineTokenize, alpha=0.01, steps=1000))
             # for logistic regression we need 2 integer labels
             intLabel = 0
             if label == 'Related and informative':
                 # if a tweet is relevant has the value 1 as label
                 intLabel = 1
             training_label.append(intLabel)

     # same operations to build testing set vectors
     for key, singleDataset in InputDatasetsTesting.items():
         for line, label in zip(singleDataset["Text"], singleDataset["Informativeness"]):
             tweetLineTokenize = line.strip().split()
             test_set.append(model.infer_vector(tweetLineTokenize, alpha=0.01, steps=1000))
             intLabel = 0
             if label == 'Related and informative':
                 intLabel = 1
             test_label.append(intLabel)

     #0.7441520467836257
     classifier = LogisticRegression()
     # train the classifier with the training set (vectors)
     classifier.fit(training_set, training_label)

     # test the classifier
     print(classifier.score(test_set, test_label))
