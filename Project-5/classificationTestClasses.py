# classificationTestClasses.py
# ----------------------------
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


from hashlib import sha1
import testClasses
# import json

from collections import defaultdict
from pprint import PrettyPrinter
pp = PrettyPrinter()

# from game import Agent
from pacman import GameState
# from ghostAgents import RandomGhost, DirectionalGhost
import random, math, traceback, sys, os
# import layout, pacman
# import autograder
# import grading

import dataClassifier, samples

VERBOSE = False



# Data sets
# ---------

EVAL_MULTIPLE_CHOICE=True

numTraining = 100
TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def readDigitData(trainingSize=100, testSize=100):
    rootdata = 'digitdata/'
    # loading digits data
    rawTrainingData = samples.loadDataFile(rootdata + 'trainingimages', trainingSize,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile(rootdata + "traininglabels", trainingSize)
    rawValidationData = samples.loadDataFile(rootdata + "validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile(rootdata + "validationlabels", TEST_SET_SIZE)
    rawTestData = samples.loadDataFile("digitdata/testimages", testSize,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", testSize)
    try:
        print "Extracting features..."
        featureFunction = dataClassifier.basicFeatureExtractorDigit
        trainingData = map(featureFunction, rawTrainingData)
        validationData = map(featureFunction, rawValidationData)
        testData = map(featureFunction, rawTestData)
    except:
        display("An exception was raised while extracting basic features: \n %s" % getExceptionTraceBack())
    return (trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData)

def readSuicideData(trainingSize=100, testSize=100):
    rootdata = 'pacmandata'
    rawTrainingData, trainingLabels = samples.loadPacmanData(rootdata + '/suicide_training.pkl', trainingSize)
    rawValidationData, validationLabels = samples.loadPacmanData(rootdata + '/suicide_validation.pkl', testSize)
    rawTestData, testLabels = samples.loadPacmanData(rootdata + '/suicide_test.pkl', testSize)
    trainingData = []
    validationData = []
    testData = []
    return (trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData)

def readContestData(trainingSize=100, testSize=100):
    rootdata = 'pacmandata'
    rawTrainingData, trainingLabels = samples.loadPacmanData(rootdata + '/contest_training.pkl', trainingSize)
    rawValidationData, validationLabels = samples.loadPacmanData(rootdata + '/contest_validation.pkl', testSize)
    rawTestData, testLabels = samples.loadPacmanData(rootdata + '/contest_test.pkl', testSize)
    trainingData = []
    validationData = []
    testData = []
    return (trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData)


smallDigitData = readDigitData(20)
bigDigitData = readDigitData(1000)

suicideData = readSuicideData(1000)
contestData = readContestData(1000)

def tinyDataSet():
    def count(m,b,h):
        c = util.Counter();
        c['m'] = m;
        c['b'] = b;
        c['h'] = h;
        return c;

    training = [count(0,0,0), count(1,0,0), count(1,1,0), count(0,1,1), count(1,0,1), count(1,1,1)]
    trainingLabels = [1,        1,            1           , 1           ,      -1     ,      -1]

    validation = [count(1,0,1)]
    validationLabels = [ 1]

    test = [count(1,0,1)]
    testLabels = [-1]

    return (training,trainingLabels,validation,validationLabels,test,testLabels);


def tinyDataSetPeceptronAndMira():
    def count(m,b,h):
        c = util.Counter();
        c['m'] = m;
        c['b'] = b;
        c['h'] = h;
        return c;

    training = [count(1,0,0), count(1,1,0), count(0,1,1), count(1,0,1), count(1,1,1)]
    trainingLabels = [1,            1,            1,          -1      ,      -1]

    validation = [count(1,0,1)]
    validationLabels = [ 1]

    test = [count(1,0,1)]
    testLabels = [-1]

    return (training,trainingLabels,validation,validationLabels,test,testLabels);


DATASETS = {
    "smallDigitData": lambda: smallDigitData,
    "bigDigitData": lambda: bigDigitData,
    "tinyDataSet": tinyDataSet,
    "tinyDataSetPeceptronAndMira": tinyDataSetPeceptronAndMira,
    "suicideData": lambda: suicideData,
    "contestData": lambda: contestData
}

DATASETS_LEGAL_LABELS = {
    "smallDigitData": range(10),
    "bigDigitData": range(10),
    "tinyDataSet": [-1,1],
    "tinyDataSetPeceptronAndMira": [-1,1],
    "suicideData": ["EAST", 'WEST', 'NORTH', 'SOUTH', 'STOP'],
    "contestData": ["EAST", 'WEST', 'NORTH', 'SOUTH', 'STOP']
}


# Test classes
# ------------

def getAccuracy(data, classifier, featureFunction=dataClassifier.basicFeatureExtractorDigit):
    trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData = data
    if featureFunction != dataClassifier.basicFeatureExtractorDigit:
        trainingData = map(featureFunction, rawTrainingData)
        validationData = map(featureFunction, rawValidationData)
        testData = map(featureFunction, rawTestData)
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    acc = 100.0 * correct / len(testLabels)
    serialized_guesses = ", ".join([str(guesses[i]) for i in range(len(testLabels))])
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (acc)
    return acc, serialized_guesses


class GradeClassifierTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GradeClassifierTest, self).__init__(question, testDict)

        self.classifierModule = testDict['classifierModule']
        self.classifierClass = testDict['classifierClass']
        self.datasetName = testDict['datasetName']

        self.accuracyScale = int(testDict['accuracyScale'])
        self.accuracyThresholds = [int(s) for s in testDict.get('accuracyThresholds','').split()]
        self.exactOutput = testDict['exactOutput'].lower() == "true"

        self.automaticTuning = testDict['automaticTuning'].lower() == "true" if 'automaticTuning' in testDict else None
        self.max_iterations = int(testDict['max_iterations']) if 'max_iterations' in testDict else None
        self.featureFunction = testDict['featureFunction'] if 'featureFunction' in testDict else 'basicFeatureExtractorDigit'

        self.maxPoints = len(self.accuracyThresholds) * self.accuracyScale


    def grade_classifier(self, moduleDict):
        featureFunction = getattr(dataClassifier, self.featureFunction)
        data = DATASETS[self.datasetName]()
        legalLabels = DATASETS_LEGAL_LABELS[self.datasetName]

        classifierClass = getattr(moduleDict[self.classifierModule], self.classifierClass)

        if self.max_iterations != None:
            classifier = classifierClass(legalLabels, self.max_iterations)
        else:
            classifier = classifierClass(legalLabels)

        if self.automaticTuning != None:
            classifier.automaticTuning = self.automaticTuning

        return getAccuracy(data, classifier, featureFunction=featureFunction)


    def execute(self, grades, moduleDict, solutionDict):
        accuracy, guesses = self.grade_classifier(moduleDict)

        # Either grade them on the accuracy of their classifer,
        # or their exact
        if self.exactOutput:
            gold_guesses = solutionDict['guesses']
            if guesses == gold_guesses:
                totalPoints = self.maxPoints
            else:
                self.addMessage("Incorrect classification after training:")
                self.addMessage("  student classifications: " + guesses)
                self.addMessage("  correct classifications: " + gold_guesses)
                totalPoints = 0
        else:
            # Grade accuracy
            totalPoints = 0
            for threshold in self.accuracyThresholds:
                if accuracy >= threshold:
                    totalPoints += self.accuracyScale

            # Print grading schedule
            self.addMessage("%s correct (%s of %s points)" % (accuracy, totalPoints, self.maxPoints))
            self.addMessage("    Grading scheme:")
            self.addMessage("     < %s:  0 points" % (self.accuracyThresholds[0],))
            for idx, threshold in enumerate(self.accuracyThresholds):
                self.addMessage("    >= %s:  %s points" % (threshold, (idx+1)*self.accuracyScale))

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        if self.exactOutput:
            _, guesses = self.grade_classifier(moduleDict)
            handle.write('guesses: "%s"' % (guesses,))

        handle.close()
        return True




class MultipleChoiceTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(MultipleChoiceTest, self).__init__(question, testDict)
        self.ans = testDict['result']
        self.question = testDict['question']

    def execute(self, grades, moduleDict, solutionDict):
        studentSolution = str(getattr(moduleDict['answers'], self.question)())
        encryptedSolution = sha1(studentSolution.strip().lower()).hexdigest()
        if encryptedSolution == self.ans:
            return self.testPass(grades)
        else:
            self.addMessage("Solution is not correct.")
            self.addMessage("Student solution: %s" % studentSolution)
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# File intentionally blank.\n')
        handle.close()
        return True


