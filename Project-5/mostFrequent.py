# mostFrequent.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
import classificationMethod

class MostFrequentClassifier(classificationMethod.ClassificationMethod):
    """
    The MostFrequentClassifier is a very simple classifier: for
    every test instance presented to it, the classifier returns
    the label that was seen most often in the training data.
    """
    def __init__(self, legalLabels):
        self.guess = None
        self.type = "mostfrequent"

    def train(self, data, labels, validationData, validationLabels):
        """
        Find the most common label in the training data.
        """
        counter = util.Counter()
        counter.incrementAll(labels, 1)
        self.guess = counter.argMax()

    def classify(self, testData):
        """
        Classify all test data as the most common label.
        """
        return [self.guess for i in testData]
