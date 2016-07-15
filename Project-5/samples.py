# samples.py
# ----------
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

## Constants
DATUM_WIDTH = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

## Module Classes

class Datum:
    """
    A datum is a pixel-level encoding of digits or face/non-face edge maps.

    Digits are from the MNIST dataset and face images are from the
    easy-faces and background categories of the Caltech 101 dataset.


    Each digit is 28x28 pixels, and each face/non-face image is 60x74
    pixels, each pixel can take the following values:
      0: no edge (blank)
      1: gray pixel (+) [used for digits only]
      2: edge [for face] or black pixel [for digit] (#)

    Pixel data is stored in the 2-dimensional array pixels, which
    maps to pixels on a plane according to standard euclidean axes
    with the first dimension denoting the horizontal and the second
    the vertical coordinate:

      28 # # # #      #  #
      27 # # # #      #  #
       .
       .
       .
       3 # # + #      #  #
       2 # # # #      #  #
       1 # # # #      #  #
       0 # # # #      #  #
         0 1 2 3 ... 27 28

    For example, the + in the above diagram is stored in pixels[2][3], or
    more generally pixels[column][row].

    The contents of the representation can be accessed directly
    via the getPixel and getPixels methods.
    """
    def __init__(self, data,width,height):
        """
        Create a new datum from file input (standard MNIST encoding).
        """
        DATUM_HEIGHT = height
        DATUM_WIDTH=width
        self.height = DATUM_HEIGHT
        self.width = DATUM_WIDTH
        if data == None:
            data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)]
        self.pixels = util.arrayInvert(convertToInteger(data))

    def getPixel(self, column, row):
        """
        Returns the value of the pixel at column, row as 0, or 1.
        """
        return self.pixels[column][row]

    def getPixels(self):
        """
        Returns all pixels as a list of lists.
        """
        return self.pixels

    def getAsciiString(self):
        """
        Renders the data item as an ascii image.
        """
        rows = []
        data = util.arrayInvert(self.pixels)
        for row in data:
            ascii = map(asciiGrayscaleConversionFunction, row)
            rows.append( "".join(ascii) )
        return "\n".join(rows)

    def __str__(self):
        return self.getAsciiString()



# Data processing, cleanup and display functions

def loadDataFile(filename, n,width,height):
    """
    Reads n data images from a file and returns a list of Datum objects.

    (Return less then n items if the end of file is encountered).
    """
    DATUM_WIDTH=width
    DATUM_HEIGHT=height
    fin = readlines(filename)
    fin.reverse()
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        if len(data[0]) < DATUM_WIDTH-1:
            # we encountered end of file...
            print "Truncating at %d examples (maximum)" % i
            break
        items.append(Datum(data,DATUM_WIDTH,DATUM_HEIGHT))
    return items

import zipfile
import os
def readlines(filename):
    "Opens a file or reads it from the zip archive data.zip"
    if(os.path.exists(filename)):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return z.read(filename).split('\n')

def loadLabelsFile(filename, n):
    """
    Reads n labels from a file and returns a list of integers.
    """
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels

def loadPacmanStatesFile(filename, n):
    f = open(filename, 'r')
    result = cPickle.load(f)
    f.close()
    return result

import cPickle
import pacmanAgents
import ghostAgents
import textDisplay
from pacman import ClassicGameRules, GameState
def loadPacmanData(filename, n):
    """
    Return game states from specified recorded games as data, and actions taken as labels
    """
    components = loadPacmanStatesFile(filename, n)
    return components['states'][:n], components['actions'][:n]

def asciiGrayscaleConversionFunction(value):
    """
    Helper function for display purposes.
    """
    if(value == 0):
        return ' '
    elif(value == 1):
        return '+'
    elif(value == 2):
        return '#'

def IntegerConversionFunction(character):
    """
    Helper function for file reading.
    """
    if(character == ' '):
        return 0
    elif(character == '+'):
        return 1
    elif(character == '#'):
        return 2

def convertToInteger(data):
    """
    Helper function for file reading.
    """
    if type(data) != type([]):
        return IntegerConversionFunction(data)
    else:
        return map(convertToInteger, data)

# Testing

def _test():
    import doctest
    doctest.testmod() # Test the interactive sessions in function comments
    n = 1
#  items = loadDataFile("facedata/facedatatrain", n,60,70)
#  labels = loadLabelsFile("facedata/facedatatrainlabels", n)
    items = loadDataFile("digitdata/trainingimages", n,28,28)
    labels = loadLabelsFile("digitdata/traininglabels", n)
    for i in range(1):
        print items[i]
        print items[i]
        print (items[i].height)
        print (items[i].width)
        print dir(items[i])
        print items[i].getPixels()

if __name__ == "__main__":
    _test()
