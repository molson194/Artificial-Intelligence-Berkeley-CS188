# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return float("inf")

        for ghostState in newGhostStates:
            if util.manhattanDistance(ghostState.getPosition(), newPos) < 2:
                return float("-inf")

        foodDist = []
        for food in list(newFood.asList()):
            foodDist.append(util.manhattanDistance(food, newPos))

        foodSuccessor = 0
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            foodSuccessor = 300

        return successorGameState.getScore() - 5 * min(foodDist) + foodSuccessor

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        maxValue = float("-inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
        return maxAction

    def getValue(self, gameState, currentDepth, agentIndex):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState,currentDepth)
        else:
            return self.minValue(gameState,currentDepth,agentIndex)

    def maxValue(self, gameState, currentDepth):
        maxValue = float("-inf")
        for action in gameState.getLegalActions(0):
            maxValue = max(maxValue, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))
        return maxValue

    def minValue(self, gameState, currentDepth, agentIndex):
        minValue = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents()-1:
                minValue = min(minValue, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth+1, 0))
            else:
                minValue = min(minValue, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1))
        return minValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        maxValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1, alpha, beta)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
            alpha = max(alpha, maxValue)
        return maxAction

    def getValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState,currentDepth,alpha,beta)
        else:
            return self.minValue(gameState,currentDepth,agentIndex,alpha,beta)

    def maxValue(self, gameState, currentDepth, alpha, beta):
        maxValue = float("-inf")
        for action in gameState.getLegalActions(0):
            maxValue = max(maxValue, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta))
            if maxValue > beta:
                return maxValue
            alpha = max(alpha, maxValue)
        return maxValue

    def minValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        minValue = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents()-1:
                minValue = min(minValue, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth+1, 0, alpha, beta))
            else:
                minValue = min(minValue, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1, alpha, beta))
            if minValue < alpha:
                return minValue
            beta = min(beta, minValue)
        return minValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        maxValue = float("-inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
        return maxAction

    def getValue(self, gameState, currentDepth, agentIndex):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState,currentDepth)
        else:
            return self.avgValue(gameState,currentDepth,agentIndex)

    def maxValue(self, gameState, currentDepth):
        maxValue = float("-inf")
        for action in gameState.getLegalActions(0):
            maxValue = max(maxValue, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))
        return maxValue

    def avgValue(self, gameState, currentDepth, agentIndex):
        avgValue = 0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents()-1:
                avgValue = avgValue + self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth+1, 0)
            else:
                avgValue = avgValue + self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1)
        return avgValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    score = scoreEvaluationFunction(currentGameState)
    newFood = currentGameState.getFood()
    newPos = currentGameState.getPacmanPosition()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    ghostDist = []
    for i in range(1, currentGameState.getNumAgents()):
        ghostDist.append(util.manhattanDistance(currentGameState.getGhostPosition(i), newPos))
    if min(ghostDist) < 2:
        return float("-inf")

    foodDist = []
    for food in list(newFood.asList()):
        foodDist.append(util.manhattanDistance(food, newPos))

    return score - 2*min(foodDist) - max(foodDist) - 8*currentGameState.getNumFood() + 1.5*min(ghostDist) + max(ghostDist)



# Abbreviation
better = betterEvaluationFunction
