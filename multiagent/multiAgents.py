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
from unittest import BaseTestSuite
from util import manhattanDistance
from game import Directions
import random, util
import math

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
    def minimax(self, gamestate, current_depth, agent_index):
        #This is if the next agent is not a ghost but again pacmans turn - increment depth
        if agent_index >= gamestate.getNumAgents():
            return self.minimax(gamestate, current_depth + 1, 0)
        
        if current_depth > self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate)
        
       
        legal_actions = []
        for action in gamestate.getLegalActions(agent_index):
            legal_actions.append(action)

        scores = []
        for action in legal_actions:
            succesor = gamestate.generateSuccessor(agent_index, action)
            score = self.minimax(succesor, current_depth, agent_index + 1)
            scores.append(score)
        
        if agent_index == 0:
            #Pacman move
            best_score = max(scores)
            if current_depth == 1:
                #This is the very first move for pacman
                best_indexes = []
                for index in range(len(scores)):
                    if scores[index] == best_score:
                        best_indexes.append(index)
                #Simply choose the first one if there are several moves that leads to the best score
                return legal_actions[best_indexes[0]]
            return best_score
        else:
            return min(scores)


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. """
        return self.minimax(gameState, 1, 0)
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def minimax_ab(self, gamestate, current_depth, agent_index, alpha, beta):
        if agent_index >= gamestate.getNumAgents():
            return self.minimax_ab(gamestate, current_depth + 1, 0, alpha, beta)
        
        if current_depth > self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate)
        
        legal_actions = []
        for action in gamestate.getLegalActions(agent_index):
            legal_actions.append(action)
        
        if agent_index == 0:
            if current_depth == 1:
                #Pacman first move
                scores = []
                for action in legal_actions:
                    succesor = gamestate.generateSuccessor(agent_index, action)
                    score = self.minimax_ab(succesor, current_depth, agent_index + 1, alpha, beta)
                    scores.append(score)

                best_score = max(scores)
                best_indexes = []
                for index in range(len(scores)):
                    if scores[index] == best_score:
                        best_indexes.append(index)
                #Simply take the first best move if several
                best_move = legal_actions[best_indexes[0]]
                return best_move

            best_score = -math.inf
            for action in legal_actions:
                succesor = gamestate.generateSuccessor(agent_index, action)
                best_score = max(best_score, self.minimax_ab(succesor, current_depth, agent_index + 1, alpha, beta))
                if best_score >= beta:
                    return best_score
                alpha = max(alpha, best_score)
            return best_score
        else:
            best_score = math.inf
            for action in legal_actions:
                succesor = gamestate.generateSuccessor(agent_index, action)
                best_score = min(best_score, self.minimax_ab(succesor, current_depth, agent_index + 1, alpha, beta))
                if best_score <= alpha:
                    return best_score
                beta = min(best_score, beta)
            return best_score

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -math.inf
        beta = math.inf
        return self.minimax_ab(gameState, 1, 0, alpha, beta)

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
