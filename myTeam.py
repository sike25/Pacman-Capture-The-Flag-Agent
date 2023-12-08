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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game, capture
import sys
from util import nearestPoint
from baselineTeam import DefensiveReflexAgent

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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'blueSikeAgent', second = 'blueAliAgent'):
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
  # if isRed:
  #   return [eval(redSike)(firstIndex), eval(redAli)(secondIndex)]
  # else:
  #   return [eval(blueSike)(firstIndex), eval(blueAli)(secondIndex)]
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


#########################################################################
class blueSikeAgent(CaptureAgent):

  def registerInitialState(self, gameState: capture.GameState):
    self.start = gameState.getAgentPosition(self.index)
    self.initialFoodCount = len(gameState.getRedFood().asList())
    self.walls = gameState.getWalls().asList()
    self.opponents = self.getOpponents(gameState)
    self.foodEaten = 0
    self.gridLength = gameState.getWalls().width
    self.gridHeight = gameState.getWalls().height
    self.path = self.pathToBoundary(gameState, self.gridLength/2, self.gridHeight/2)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState: capture.GameState):
    # go to the other side at the start of the game
    while len(self.path) > 0:
      return self.path.pop(0)
    
    # choose best action (driven by self.evaluate)
    actions = gameState.getLegalActions(self.index)
    actions.remove("Stop")
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    bestAction = random.choice(bestActions)

    # updates the food count as we eating
    sa, sb = self.getSuccessor(gameState, bestAction).getAgentPosition(self.index)
    if gameState.hasFood(sa, sb):
      self.foodEaten += 1
    # resets food count after we deposit on our side
    if gameState.getAgentPosition(self.index)[0] < self.gridLength/2:
      if sa > self.gridLength/2:
        self.foodEaten = 0
    return bestAction

  def evaluate(self, gameState: capture.GameState, action):
    sa, sb = self.getSuccessor(gameState, action).getAgentPosition(self.index)
    opponentsPosition = []
    opponentsPosition.append(gameState.getAgentPosition(self.opponents[0]))
    opponentsPosition.append(gameState.getAgentPosition(self.opponents[1]))
    foodToEat = self.getFood(gameState).asList()

    if self.getMazeDistance(opponentsPosition[0], (sa, sb)) <= 1 or self.getMazeDistance(opponentsPosition[1], (sa, sb)) <= 1:
      return -sys.maxsize

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights


  
    if self.isCorner(gameState, sa, sb) and self.getMazeDistance((sa, sb), opponentsPosition[0]) <= 5:
      return -sys.maxsize
    if self.isCorner(gameState, sa, sb) and self.getMazeDistance((sa, sb), opponentsPosition[1]) <= 5:
      return -sys.maxsize
    if self.foodEaten >= 5:
      return 1/(self.getNearestDistanceHome(gameState, sa, sb) + 1)
    if gameState.hasFood(sa, sb):
      return 1
    else:
      return 1/(self.getDistanceToNearestFood(gameState, sa, sb, foodToEat) + 1)
    


  def getFeatures(self, gameState: capture.GameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myPos = successor.getAgentState(self.index).getPosition()

    # Compute my new score
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Compute nearest distance to enemies - closer and further
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    attackers = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(attackers) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in attackers]
      features['minAttackerDistance'] = min(dists)
      if len(attackers) > 1:
        features['maxAttackerDistance'] = max(dists)

    # Compute distance to nearest capsule
    enemyCapsulesDistances = [self.getMazeDistance(myPos, capsule) for capsule in gameState.getRedCapsules()]
    if len(enemyCapsulesDistances) > 0:
      features['nearestEnemyCapsule'] = min(enemyCapsulesDistances)

    # Compute distance from home
    features['nearestDistanceFromHome'] = self.getNearestDistanceHome(gameState, myPos[0], myPos[1])
    return features

    # Special cases
      # weight of distance from home only become relevant after eating some no of food 
      # weight of nearest capsule increases as distance from enemy reduces
      # right after eating capsule, the next * moves reverses the weight of distance from enemy


  def getWeights(self, gameState: capture.GameState, action):
    homeWeight = 0
    if self.foodEaten >= 1:
      homeWeight = 1000
    if len(gameState.getRedCapsules()) == 0:
      capsuleDistanceWeight
    else:
      capsuleDistanceWeight = 1/self.getFeatures(gameState, action)['minAttackerDistance']
    return {'successorScore': 9999, 'distanceToFood': -20, 'minAttackerDistance': 10, 'maxAttackerDistance': 1, 'nearestEnemyCapsule': capsuleDistanceWeight, 'nearestDistanceFromHome': homeWeight }


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
    
  def getNearestDistanceHome(self, gameState: capture.GameState, sa, sb):
    x = int(self.gridLength / 2)
    distances = []
    for y in range(self.gridLength):
      if self.inGrid(x, y):
        distances.append(self.getMazeDistance((sa, sb), (x, y)))
    return min(distances)
  
  def inGrid(self, x, y):
    try:
      self.getMazeDistance((30, 12), (x, y))
      return True
    except Exception:
      return False

  def getDistanceToNearestFood(self, gameState: capture.GameState, sa, sb, foodToEat):
    distances = []
    for food in foodToEat:
      distances.append(self.getMazeDistance((sa, sb), food))
    return min(distances)

  def isCorner(self, gameState: capture.GameState, x, y):
    around = [(x + 1, y + 1), (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1)]
    wallCount = 0
    for a, b in around:
      if gameState.hasWall(a, b):
        wallCount += 1
    return wallCount >= 3
  
  
  def pathToBoundary(self, gameState: capture.GameState, x, y):
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
        if presentState[0] == x and presentState[1] <= y:
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


################################################################################################



############################################################################
class blueAliAgent(CaptureAgent):

  def registerInitialState(self, gameState: capture.GameState):
    self.start = gameState.getAgentPosition(self.index)
    self.initialFoodCount = len(gameState.getRedFood().asList())
    self.walls = gameState.getWalls().asList()
    self.opponents = self.getOpponents(gameState)
    self.foodEaten = 0
    self.gridLength = gameState.getWalls().width
    self.gridHeight = gameState.getWalls().height
    self.path = self.pathToBoundary(gameState, self.gridLength/2, self.gridHeight/2)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState: capture.GameState):
    # go to the other side at the start of the game
    while len(self.path) > 0:
      return self.path.pop(0)
    
    # choose best action (driven by self.evaluate)
    actions = gameState.getLegalActions(self.index)
    actions.remove("Stop")
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    bestAction = random.choice(bestActions)

    # updates the food count as we eating
    sa, sb = self.getSuccessor(gameState, bestAction).getAgentPosition(self.index)
    if gameState.hasFood(sa, sb):
      self.foodEaten += 1
    # resets food count after we deposit on our side
    if gameState.getAgentPosition(self.index)[0] < self.gridLength/2:
      if sa > self.gridLength/2:
        self.foodEaten = 0
    return bestAction
  

  def evaluate(self, gameState: capture.GameState, action):
    sa, sb = self.getSuccessor(gameState, action).getAgentPosition(self.index)
    opponentsPosition = []
    opponentsPosition.append(gameState.getAgentPosition(self.opponents[0]))
    opponentsPosition.append(gameState.getAgentPosition(self.opponents[1]))
    foodToEat = self.getFood(gameState).asList()

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

    if self.getMazeDistance(opponentsPosition[0], (sa, sb)) <= 3 or self.getMazeDistance(opponentsPosition[1], (sa, sb)) <= 5:
      return -sys.maxsize
    if self.isCorner(gameState, sa, sb) and self.getMazeDistance((sa, sb), opponentsPosition[0]) <= 5:
      return -sys.maxsize
    if self.isCorner(gameState, sa, sb) and self.getMazeDistance((sa, sb), opponentsPosition[1]) <= 5:
      return -sys.maxsize
    if self.foodEaten >= self.initialFoodCount/8:
      return 1/(self.getNearestDistanceHome(gameState, sa, sb) + 1)
    if gameState.hasFood(sa, sb):
      return 1
    else:
      return 1/(self.getDistanceToNearestFood(gameState, sa, sb, foodToEat) + 1)
    

  def getFeatures(self, gameState: capture.GameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myPos = successor.getAgentState(self.index).getPosition()

    # Compute my new score
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Compute nearest distance to enemies - closer and further
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    attackers = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(attackers) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in attackers]
      features['minAttackerDistance'] = min(dists)
      if len(attackers) > 1:
        features['maxAttackerDistance'] = max(dists)

    # Compute distance to nearest capsule
    enemyCapsulesDistances = [self.getMazeDistance(myPos, capsule) for capsule in gameState.getRedCapsules()]
    if len(enemyCapsulesDistances) > 0:
      features['nearestEnemyCapsule'] = min(enemyCapsulesDistances)

    # Compute distance from home
    features['nearestDistanceFromHome'] = self.getNearestDistanceHome(gameState, myPos[0], myPos[1])
    return features

    # Special cases
      # weight of distance from home only become relevant after eating some no of food 
      # weight of nearest capsule increases as distance from enemy reduces
      # right after eating capsule, the next * moves reverses the weight of distance from enemy


  def getWeights(self, gameState: capture.GameState, action):
    homeWeight = 0
    if self.foodEaten >= 1:
      homeWeight = 1000

    if len(gameState.getRedCapsules()):
      capsuleDistanceWeight = 0
    else:
      capsuleDistanceWeight = 1/self.getFeatures(gameState, action)['minAttackerDistance']
    return {'successorScore': 9999, 'distanceToFood': -200, 'minAttackerDistance': 10, 'maxAttackerDistance': 1, 'nearestEnemyCapsule': capsuleDistanceWeight, 'nearestDistanceFromHome': homeWeight }

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
    
  def getNearestDistanceHome(self, gameState: capture.GameState, sa, sb):
    x = int(self.gridLength / 2)
    distances = []
    for y in range(self.gridLength):
      if self.inGrid(x, y):
        distances.append(self.getMazeDistance((sa, sb), (x, y)))
    return min(distances)
  
  def inGrid(self, x, y):
    try:
      self.getMazeDistance((30, 12), (x, y))
      return True
    except Exception:
      return False

  def getDistanceToNearestFood(self, gameState: capture.GameState, sa, sb, foodToEat):
    distances = []
    for food in foodToEat:
      distances.append(self.getMazeDistance((sa, sb), food))
    return min(distances)

  def isCorner(self, gameState: capture.GameState, x, y):
    around = [(x + 1, y + 1), (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1)]
    wallCount = 0
    for a, b in around:
      if gameState.hasWall(a, b):
        wallCount += 1
    return wallCount >= 3
  


  
  def pathToBoundary(self, gameState: capture.GameState, x, y):
    """Finds the shortest path to the opponent's side,
    and spends the first part of the game just going there
    """
    frontier = util.Queue() #FIFO
    frontier.push(Node(self.start, 0, None, None))
    explored = set() 
    while True:
        if frontier.isEmpty():
            return []    
        presentNode = frontier.pop() # chooses the shallowest node.
        presentState = presentNode.getState()
        if presentState[0] == x and presentState[1] > y:
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
  


################################################################################################


  
class redAli(CaptureAgent):
  def registerInitialState(self, gameState: capture.GameState):
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState: capture.GameState):
    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)
  
class blueSike(CaptureAgent):
  def registerInitialState(self, gameState: capture.GameState):
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState: capture.GameState):
    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)

  
class blueAli(CaptureAgent):
  def registerInitialState(self, gameState: capture.GameState):
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState: capture.GameState):
    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)
    

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

