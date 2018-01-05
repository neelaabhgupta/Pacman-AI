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
        fdist = []
        gdist = 0
        for x in range(0,newFood.width):
            for y in range(0, newFood.height):
                if newFood[x][y]:
                    fdist.append(util.manhattanDistance(newPos, (x, y)))
        newGhostStates = successorGameState.getGhostStates()
        ghostPosition = successorGameState.getGhostPositions()
        gdist = util.manhattanDistance(newPos, ghostPosition[0])
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        if fdist:
            if gdist > 1:
                return 1.0/gdist + 1.0/min(fdist) + successorGameState.getScore()
        else:
            return successorGameState.getScore()


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
        "*** YOUR CODE HERE ***"

        score, move = self.minimax(gameState, 0, 0, True)
        return move

    def minimax(self, gameState, agent, depth, maximizingPlayer):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), 0

        if maximizingPlayer:
            moves = gameState.getLegalActions(agent)
            bestValue = (float("-inf"))
            for i in moves:
                successor = gameState.generateSuccessor(agent, i)
                successorScore, successorMove = self.minimax(successor, 1, depth, False)
                if successorScore > bestValue:
                    bestValue = successorScore
                    bestMove = i
            return bestValue, bestMove

        else:
            moves = gameState.getLegalActions(agent)
            bestValue = (float("inf"))
            for i in moves:
                if agent == gameState.getNumAgents() - 1:
                    successor = gameState.generateSuccessor(agent, i)
                    successorScore, successorMove = self.minimax(successor, 0, depth + 1, True)
                else:
                    successor = gameState.generateSuccessor(agent, i)
                    successorScore, successorMove = self.minimax(successor, agent + 1, depth, False)
                if successorScore < bestValue:
                    bestValue = successorScore
                    worstMove = i
            return bestValue, worstMove

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score, move = self.alphabeta(gameState, 0, 0, float("-inf"), float("inf"), True)
        return move

    def alphabeta(self, gameState, agent, depth, alpha, beta, maximizingPlayer):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), 0

        if maximizingPlayer:
            moves = gameState.getLegalActions(agent)
            bestValue = (float("-inf"))
            for i in moves:
                successor = gameState.generateSuccessor(agent, i)
                successorScore, successorMove = self.alphabeta(successor, 1, depth, alpha, beta, False)
                if successorScore > bestValue:
                    bestValue = successorScore
                    bestMove = i
                if bestValue > beta:
                    return bestValue, bestMove
                alpha = max(alpha, bestValue)
            return bestValue, bestMove

        else:
            moves = gameState.getLegalActions(agent)
            bestValue = (float("inf"))
            for i in moves:
                if agent == gameState.getNumAgents() - 1:
                    successor = gameState.generateSuccessor(agent, i)
                    successorScore, successorMove = self.alphabeta(successor, 0, depth + 1, alpha, beta, True)
                else:
                    successor = gameState.generateSuccessor(agent, i)
                    successorScore, successorMove = self.alphabeta(successor, agent + 1, depth, alpha, beta, False)
                if successorScore < bestValue:
                    bestValue = successorScore
                    worstMove = i
                if bestValue < alpha:
                    return  bestValue, worstMove
                beta = min(beta, bestValue)
            return bestValue, worstMove
        util.raiseNotDefined()

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
        score, move = self.expmax(gameState, 0, 0, True)
        return move

    def expmax(self, gameState, agent, depth, maximizingPlayer):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), 0

        if maximizingPlayer:
            moves = gameState.getLegalActions(agent)
            bestValue = (float("-inf"))
            for i in moves:
                successor = gameState.generateSuccessor(agent, i)
                successorScore, successorMove = self.expmax(successor, 1, depth, False)
                if successorScore > bestValue:
                    bestValue = successorScore
                    bestMove = i
            return bestValue, bestMove

        else:
            moves = gameState.getLegalActions(agent)
            bestValue = 0
            sum = 0
            avg = 0.0
            for i in moves:
                if agent == gameState.getNumAgents() - 1:
                    successor = gameState.generateSuccessor(agent, i)
                    successorScore, successorMove = self.expmax(successor, 0, depth + 1, True)
                else:
                    successor = gameState.generateSuccessor(agent, i)
                    successorScore, successorMove = self.expmax(successor, agent + 1, depth, False)

                sum = sum + successorScore
            avg = sum/float(len(moves))
            bestValue = bestValue + avg
            return bestValue, 0
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

