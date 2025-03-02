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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        score = successorGameState.getScore()            

        foodList = newFood.asList()
        if foodList:
            closest = min(manhattanDistance(newPos, pos) for pos in foodList)
            score += 5 / (closest + 1)

        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            distance = manhattanDistance(newPos, ghost.getPosition())
            if scaredTime > 0:
                score += 5 / (distance + 1)
            if ghost.scaredTimer == 0 and distance < 2:
                score -= 10 / (distance + 1)

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maximize(gameState, 1)
    
    def maximize(self, gameState, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        max = float("inf") * -1
        ans = Directions.STOP
        # Note: map each move into game state for pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            current = self.minimize(successor, depth, 1)
            if current > max:
                max = current
                ans = action
        if depth > 1:
            return max
        
        return ans
            
    def minimize(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        mini = float("inf")
        # Note: map each move into game state for current ghost
        successors = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)]
        if agent == gameState.getNumAgents() - 1: # Note: if equal means that we are on the last ghost
            if depth < self.depth: # Note: we have to go deeper at least one level
                for successor in successors:
                    mini = min(mini, self.maximize(successor, depth + 1))
            else: # Note: we have reach correct amount of layers
                for successor in successors:
                    mini = min(mini, self.evaluationFunction(successor))
        else: # Note: means there are more ghosts to iterate
            for successor in successors:
                mini = min(mini, self.minimize(successor, depth, agent + 1))

        return mini

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 1)
    
    def expectimax(self, gameState, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        max = float("inf") * -1
        ans = Directions.STOP
        # Note: map each move into game state for pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            current = self.expected(successor, depth, 1)
            if current > max:
                max = current
                ans = action
        if depth > 1:
            return max
        
        return ans
    
    def expected(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        avg = 0.0
        # Note: map each move into game state for current ghost and add expected value
        successors = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)]
        moves = len(successors)
        if agent == gameState.getNumAgents() - 1: # Note: if equal means that we are on the last ghost
            if depth < self.depth: # Note: we have to go deeper at least one level
                for successor in successors:
                    avg += self.expectimax(successor, depth + 1) / moves
            else: # Note: we have reach correct amount of layers
                for successor in successors:
                    avg += self.evaluationFunction(successor) / moves
        else: # Note: means there are more ghosts to iterate
            for successor in successors:
                avg += self.expected(successor, depth, agent + 1) / moves

        return avg

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Similar to the evaluation function I made in the Reflex Agent from question 1
    I plan to leverage the score with the distance from food and ghosts. I also 
    added some consideration to the power pellets in order to conserve them. For 
    the ghosts I needed to change it so that the score goes down when ghosts
    are close but also aren't scared since these aren't of threat to pac man. After
    some trial and error I want to include the number of food left on the screen
    as a negative drawback to stop the pacman from deciding that STOP is the
    optimal move at the current state.
    """
    "*** YOUR CODE HERE ***"
    curr = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    power = currentGameState.getCapsules()
    times = [ghostState.scaredTimer for ghostState in ghosts]

    score = currentGameState.getScore()           

    foodList = food.asList()
    if foodList:
        closest = min(manhattanDistance(curr, pos) for pos in foodList)
        score += 5 / (closest + 1)

    closest_ghost = float("inf")
    ghost_score = 0
    for ghost, scaredTime in zip(ghosts, times):
        distance = manhattanDistance(curr, ghost.getPosition())
        if scaredTime == 0:
            closest_ghost = min(closest_ghost, distance)
        if scaredTime > distance:
            ghost_score += 50 - distance

    return score - 5 * len(foodList) + 5 * len(power) + ghost_score
# Abbreviation
better = betterEvaluationFunction
