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


from argparse import Action
from pickle import FALSE
from sre_constants import SUCCESS
from unittest import BaseTestSuite
from util import manhattanDistance
from game import Directions
import random, util
import math
import sys

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        currentFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isLose():
            return -math.inf
        elif successorGameState.isWin():
            return math.inf
        
        
        #Distance to closest active ghost
        activeGhosts = []
        scaredGhosts = []
        for ghost in newGhostStates:
            if ghost.scaredTimer > 0:
                scaredGhosts.append(ghost)
            else:
                activeGhosts.append(ghost)

        #Sorting both lists of ghost from closest to the pacman,  based on the ghost.getPosition()
        activeGhosts.sort(key=lambda x: manhattanDistance(newPos, x.getPosition()))
        scaredGhosts.sort(key=lambda x: manhattanDistance(newPos, x.getPosition()))
       
        #Closest ghost 

        #Here there is a lot more to do, but currently i dont have the time
        #The idea is to eat the scared ghosts if doable, instead just avoid all ghost that are closer to the pacman than 3 tiles if not if 
        # ignore it
        if len(activeGhosts) > 0:
            if manhattanDistance(newPos, activeGhosts[0].getPosition()) < 2:
                return -math.inf
        
        #Distance to closest food
        foodList = currentFood.asList()
        foodList.sort(key=lambda x: manhattanDistance(newPos, x))
        if len(foodList) > 0:
            foodDistance = manhattanDistance(newPos, foodList[0])
        
        evaluation = successorGameState.getScore()/200 - foodDistance
        return evaluation

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
    def maxValue(self, agent_index, depth, gameState):
        current_depth = depth + 1
        if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
            return self.evaluationFunction(gameState)
        max_value = -math.inf
        legal_actions = gameState.getLegalActions(agent_index)
        for action in legal_actions:
            succesor = gameState.generateSuccessor(agent_index, action)
            max_value = max(max_value, self.minValue(agent_index+1, current_depth, succesor))
        return max_value

    def minValue(self, agent_index, depth, gameState):
        min_value = math.inf
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legal_actions = gameState.getLegalActions(agent_index)
        for action in legal_actions:
            successor = gameState.generateSuccessor(agent_index, action)
            if agent_index == (gameState.getNumAgents() - 1):
                min_value = min(min_value, self.maxValue(0, depth, successor))
            else:
                min_value = min(min_value, self.minValue(agent_index+1, depth, successor))
        return min_value


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. """

        #return self.minimax(gameState, 1, 0)
        legal_actions = gameState.getLegalActions(0)
        action_scores = [self.minValue(1, 0, gameState.generateSuccessor(0, action)) for action
                         in legal_actions]
        print(action_scores)
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = max_indices[-1]
        return legal_actions[chosenIndex]
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        return util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
