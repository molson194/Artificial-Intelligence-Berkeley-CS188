# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Path(object):
    def __init__(self, locations, directions, cost):
        self.locations = locations
        self.directions = directions
        self.cost = cost

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """ Search the deepest nodes in the search tree first."""

    path = Path([problem.getStartState()],[],0)

    if problem.isGoalState(problem.getStartState()):
        return path.directions

    stack = util.Stack()
    stack.push(path)

    while not stack.isEmpty():
        currentPath = stack.pop()
        currentLocation = currentPath.locations[-1]
        if problem.isGoalState(currentLocation):
            return currentPath.directions
        else:
            nextSteps = problem.getSuccessors(currentLocation)
            for nextStep in nextSteps:
                nextLocation = nextStep[0]
                nextDirection = nextStep[1]
                nextCost = nextStep[2]
                if nextLocation not in currentPath.locations:
                    nextLocations = currentPath.locations[:]
                    nextLocations.append(nextLocation)
                    nextDirections = currentPath.directions[:]
                    nextDirections.append(nextDirection)
                    nextCosts = currentPath.cost + nextCost
                    nextPath = Path(nextLocations, nextDirections, nextCosts)
                    stack.push(nextPath)

    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    path = Path([problem.getStartState()],[],0)

    if problem.isGoalState(problem.getStartState()):
        return path.directions

    queue = util.Queue()
    queue.push(path)
    visited = [problem.getStartState()]

    while not queue.isEmpty():
        currentPath = queue.pop()
        currentLocation = currentPath.locations[-1]
        if problem.isGoalState(currentLocation):
            return currentPath.directions
        else:
            nextSteps = problem.getSuccessors(currentLocation)
            for nextStep in nextSteps:
                nextLocation = nextStep[0]
                nextDirection = nextStep[1]
                nextCost = nextStep[2]
                if (nextLocation not in currentPath.locations) and (nextLocation not in visited):
                    if not problem.isGoalState(nextLocation):
                        visited.append(nextLocation)
                    nextLocations = currentPath.locations[:]
                    nextLocations.append(nextLocation)
                    nextDirections = currentPath.directions[:]
                    nextDirections.append(nextDirection)
                    nextCosts = currentPath.cost + nextCost
                    nextPath = Path(nextLocations, nextDirections, nextCosts)
                    queue.push(nextPath)

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    path = Path([problem.getStartState()],[],0)

    if problem.isGoalState(problem.getStartState()):
        return path.directions

    queue = util.PriorityQueue()
    queue.push(path, 0)
    visited = [problem.getStartState()]

    while not queue.isEmpty():
        currentPath = queue.pop()
        currentLocation = currentPath.locations[-1]
        if problem.isGoalState(currentLocation):
            return currentPath.directions
        else:
            nextSteps = problem.getSuccessors(currentLocation)
            for nextStep in nextSteps:
                nextLocation = nextStep[0]
                nextDirection = nextStep[1]
                nextCost = nextStep[2]
                if (nextLocation not in currentPath.locations) and (nextLocation not in visited):
                    if not problem.isGoalState(nextLocation):
                        visited.append(nextLocation)
                    nextLocations = currentPath.locations[:]
                    nextLocations.append(nextLocation)
                    nextDirections = currentPath.directions[:]
                    nextDirections.append(nextDirection)
                    nextCosts = currentPath.cost + nextCost
                    nextPath = Path(nextLocations, nextDirections, nextCosts)
                    queue.push(nextPath, nextCosts)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    path = Path([problem.getStartState()],[],0)

    if problem.isGoalState(problem.getStartState()):
        return path.directions

    queue = util.PriorityQueue()
    queue.push(path, 0)
    visited = [problem.getStartState()]

    while not queue.isEmpty():
        currentPath = queue.pop()
        currentLocation = currentPath.locations[-1]
        if problem.isGoalState(currentLocation):
            return currentPath.directions
        else:
            nextSteps = problem.getSuccessors(currentLocation)
            for nextStep in nextSteps:
                nextLocation = nextStep[0]
                nextDirection = nextStep[1]
                nextCost = nextStep[2]
                if (nextLocation not in currentPath.locations) and (nextLocation not in visited):
                    if not problem.isGoalState(nextLocation):
                        visited.append(nextLocation)
                    nextLocations = currentPath.locations[:]
                    nextLocations.append(nextLocation)
                    nextDirections = currentPath.directions[:]
                    nextDirections.append(nextDirection)
                    nextCosts = currentPath.cost + nextCost
                    nextHeuristic = heuristic(nextLocation, problem)
                    nextPath = Path(nextLocations, nextDirections, nextCosts)
                    queue.push(nextPath, nextCosts + nextHeuristic)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
