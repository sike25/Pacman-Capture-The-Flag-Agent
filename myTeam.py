# myTeam.py
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


from baselineTeam import DefensiveReflexAgent
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from util import nearestPoint
import game, capture
import sys

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AttackAgent', second = 'DefendAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


#########################################################################################
#########################################################################################
#########################################################################################
# START OF OFFENSE


class AttackAgent(CaptureAgent):
  def registerInitialState(self, gameState: capture.GameState):
    self.start = gameState.getAgentPosition(self.index)
    self.initialFoodCount = len(self.getFood(gameState).asList())
    self.walls = gameState.getWalls().asList()
    self.opponents = self.getOpponents(gameState)
    self.foodEaten = 0
    self.gridLength = gameState.getWalls().width
    self.gridHeight = gameState.getWalls().height
    self.depth = 2 # for running minimax
    self.team = self.getTeam(gameState)
    self.otherTeam = [2, 0] if self.team == [1, 3] else [1, 3]
    self.path = self.pathToBoundary(gameState)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState: capture.GameState):
    # at the start of the game, go to the boundary
    while len(self.path) > 0:
      return self.path.pop(0)
  
    # choose best action (driven by self.evaluate)
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    bestAction = random.choice(bestActions)
    return bestAction
  
  def evaluate(self, gameState: capture.GameState, action):
    successor = self.getSuccessor(gameState, self.index, action)
    return self.maxValue(successor, -sys.maxsize, sys.maxsize, self.index, 0)[0]

  def maxValue(self, gameState, alpha, beta, index, depth):
    # terminate
    if depth == self.depth or gameState.isOver():
      return (self.evaluateState(gameState, index), "")
        
    # find the maximum utility
    maxV = -sys.maxsize
    maxAction = ""
    for action in gameState.getLegalActions(index):  
        # call the first ghost   
        minPredictedVal = self.minValue(self.getSuccessor(gameState, index, action), alpha, beta, index+1, depth)[0]
        maxV = max(maxV, minPredictedVal)

        # update the action that resulted in the maximum value
        if minPredictedVal == maxV:
          maxAction = action

        # prune
        if maxV > beta:
          return (maxV, action)
        alpha = max(alpha, maxV)

    # maxUtility, action that results in maxUtility
    return (maxV, maxAction)

  def minValue(self, gameState, alpha, beta, index, depth):
    # terminate
    if depth == self.depth or gameState.isOver():
      return (self.evaluateState(gameState, index), "")
        
    # find the minimum utility
    minV = sys.maxsize
    minAction = ""
    for action in gameState.getLegalActions(index):
      predictedVal = 0

      # if this is the last ghost, call max
      if index == self.otherTeam[1]:
        predictedVal = self.maxValue(self.getSuccessor(gameState, index, action), alpha, beta, self.team[0], depth + 1)[0]
      # if this is not the last ghost, call the next ghost
      else:
        predictedVal = self.minValue(self.getSuccessor(gameState, index, action), alpha, beta, self.otherTeam[1], depth)[0]
      minV = min(minV, predictedVal)

        # update the action that resulted in the minimum utility
      if predictedVal == minV:
        minAction = action

      # prune
      if minV < alpha:
        return (minV, action)
      beta = min(beta, minV)

    # minUtility, action that results in minUtility
    return (minV, minAction)
  
  def evaluateState(self, state, index):
    features = self.getFeatures(state, index)
    weights = self.getWeights(state, index)
    return features * weights
  
  def getFeatures(self, state, index):
    features = util.Counter()
    myPos = state.getAgentPosition(index)

    # Compute my new score
    foodList = self.getFood(state).asList()    
    features['successorScore'] = -len(foodList) # or self.getScore(successor)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Compute nearest distance to enemies - closer and further
    enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
    attackers = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(attackers) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in attackers]
      features['minAttackerDistance'] = min(dists)
      if len(attackers) > 1:
        features['maxAttackerDistance'] = max(dists)

    # Compute distance to nearest capsule
    enemyCapsulesDistances = [self.getMazeDistance(myPos, capsule) for capsule in self.getCapsules(state)]
    if len(enemyCapsulesDistances) > 0:
      features['nearestEnemyCapsule'] = min(enemyCapsulesDistances)

    # Compute distance from home
    features['nearestDistanceFromHome'] = self.getNearestDistanceHome(state, myPos[0], myPos[1])
  
    # Compute number of eaten pellets
    features['carrying'] = state.getAgentState(self.index).numCarrying

    # Do we get eaten?
    features['eaten'] = 0
    for attacker in attackers:
      if self.getMazeDistance(myPos, attacker.getPosition()) <= 1:
        features['eaten'] = 1

    return features


    # Special cases
      # weight of distance from home only become relevant after eating some no of food
      # weight of nearest capsule increases as distance from enemy reduces
      # right after eating capsule, the next * moves reverses the weight of distance from enemy

  def getWeights(self, state, index):
    features = self.getFeatures(state, index)
    homeWeight = -features['carrying'] * 100

    if len(self.getCapsules(state)) == 0:
      capsuleDistanceWeight = 0
    else:
      capsuleDistanceWeight = -(features['minAttackerDistance']/100)

    return {'successorScore': 1000, 'distanceToFood': -10, 'minAttackerDistance': 10, 'maxAttackerDistance': 0.025, 'nearestEnemyCapsule': capsuleDistanceWeight, 'nearestDistanceFromHome': homeWeight, 'eaten': -200 }
  

  ######################################################################################
  # Helper Methods
  
  def getSuccessor(self, gameState: capture.GameState, index, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(index, action)
    pos = successor.getAgentState(index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(index, action)
    else:
      return successor
    
  def getNearestDistanceHome(self, gameState: capture.GameState, sa, sb):
    if self.red:
      x = int(self.gridLength / 2)
    else:
      x = int(self.gridLength / 2) - 1
    
    distances = []
    for y in range(self.gridLength):
      try:
        distances.append(self.getMazeDistance((sa, sb), (x, y)))
      except Exception:
        distances
    return min(distances)


    
  ###################################################################################
    

  ######################################################################################
  # Code for finding the shortest path to the boundary (starts the game)
  # Breadth First Search
  def pathToBoundary(self, gameState: capture.GameState):
    """Finds the shortest path to the opponent's side,
    and spends the first part of the game just going there
    """
    frontier = util.Queue() #FIFO
    frontier.push(Node(self.start, 0, None, None))
    explored = set()
    while True:
        if frontier.isEmpty():
            return []    
        presentNode = frontier.pop() # chooses the shallowest node
        presentState = presentNode.getState()
        if self.isBoundary(presentState[0]):
            return self.getPath(presentNode, self.start)
        if presentState not in explored:
            nextStates = self.getLegalNeighbors(presentState, gameState.getWalls())
            for nextState in nextStates:
                childNode = Node(nextState[0], 1 + presentNode.getPathCost(), nextState[1], presentNode)
                if nextState[0] not in explored:
                    frontier.push(childNode)
        explored.add(presentState)

  def getLegalNeighbors(self, position, walls):
    x,y = position
    x_int, y_int = int(x + 0.5), int(y + 0.5)
    neighbors = []
    for dir, vec in game.Actions._directionsAsList:
      dx, dy = vec
      next_x = x_int + dx
      if next_x < 0 or next_x == walls.width: continue
      next_y = y_int + dy
      if next_y < 0 or next_y == walls.height: continue
      if not walls[next_x][next_y]: neighbors.append(((next_x, next_y), dir))
    return neighbors

  def getPath(self, node, state):
    path = []
    while node.getState() != state:
        path.append(node.getLastAction())
        node = node.getParentNode()
    return path[::-1]

  def isBoundary(self, xPosition):
    if self.red:
      return xPosition == self.gridLength/2
    else:
      return xPosition == (self.gridLength/2) - 1
  
class Node():
    """
        a Node object has: a state pointer, total path cost,
        last action taken and parent Node.
        It stores a more comprehensive idea of what leads to
        a state
    """
    def __init__(self, state, pathCost, lastAction, parentNode):
        self.state = state # tuple
        self.pathCost = pathCost # int
        self.lastAction = lastAction # string
        self.parentNode = parentNode # Node
    def getState(self):
        return self.state
    def getPathCost(self):
        return self.pathCost
    def getLastAction(self):
        return self.lastAction
    def getParentNode(self):
        return self.parentNode

#########################################################################################

# END OF OFFENSE
#########################################################################################
#########################################################################################
#########################################################################################









#########################################################################################
#########################################################################################
#########################################################################################
# BASELINE DEFENCE REUSED, BECAUSE IT IS TRULY GOOD
class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState: capture.GameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState: capture.GameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState: capture.GameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState: capture.GameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState: capture.GameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class DefendAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState: capture.GameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState: capture.GameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

#########################################################################################
#########################################################################################
#########################################################################################



class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState: capture.GameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState: capture.GameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)