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
import random
import time
import util
from game import Directions
import game

import math
import queue


DEBUG = True

FIRST = 'OffensiveAgent'
SECOND = 'DefensiveAgent'

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed, first=FIRST, second=SECOND):
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


"""Helper functions"""

def bfs(pos: tuple, walls: list, goalTest) -> list:
    """Finds shortest path from pos to the goal satisfying goalTest.

    Arguments:
    - pos -- Start point
    - walls -- List of walls in the map
    - goalTest -- Lambda function for goal test

    Return:
    - List of actions
    """
    q = queue.Queue()
    current = (pos, [])
    q.put(current)
    visited = set()

    while not q.empty():
        node, path = q.get()
        if goalTest(node[0], node[1]):
            if path == []:
                path = ['Stop']
            return path
        if not node in visited:
            visited.add(node)
            neighbors = game.Actions.getLegalNeighbors(node, walls)
            for nx, ny in neighbors:
                dx, dy = nx - node[0], ny - node[1]
                action = game.Actions.vectorToDirection((dx, dy))
                q.put(((nx, ny), path + [action]))

    # return list of None when bfs fails
    return [None]


def bfsEvade(pos: tuple, walls: list, goalTest, evadeList) -> list:
    """Same as bfs, but path should not go through points in evadeList.

    Arguments:
    - pos -- Start point
    - walls -- List of walls in the map
    - goalTest -- Lambda function for goal test
    - evadeList -- List of points path should not include

    Return:
    - List of actions
    """
    q = queue.Queue()
    current = (pos, [])
    q.put(current)
    visited = set()

    while not q.empty():
        node, path = q.get()
        if goalTest(node[0], node[1]):
            if path == []:
                path = ['Stop']
            return path
        if not node in visited:
            visited.add(node)
            neighbors = game.Actions.getLegalNeighbors(node, walls)
            for nx, ny in neighbors:
                if (nx, ny) not in evadeList:
                    dx, dy = nx - node[0], ny - node[1]
                    action = game.Actions.vectorToDirection((dx, dy))
                    q.put(((nx, ny), path + [action]))

    # return list of None when bfs fails
    return [None]


def astar(pos: tuple, walls: list, goalTest, heuristic, gameState) -> list:
    """Perform astar search from pos to the goal satisfying goalTest 
    using heuristic.

    Arguments:
    - pos -- Start point
    - walls -- List of walls in the map
    - goalTest -- Lambda function for goal test
    - heuristic -- Heuristic function for calculating forward cost

    Return:
    - List of actions
    """
    pq = util.PriorityQueue()
    current = (pos, [], 0)
    pq.push(current, 0)
    visited = set()

    while not pq.isEmpty():
        node, path, priority = pq.pop()
        if goalTest(node[0], node[1]):
            return path
        if not node in visited:
            visited.add(node)
            neighbors = game.Actions.getLegalNeighbors(node, walls)
            for nx, ny in neighbors:
                dx, dy = nx - node[0], ny - node[1]
                action = game.Actions.vectorToDirection((dx, dy))
                newPriority = priority + 1 + heuristic((nx, ny), gameState)
                pq.push(((nx, ny), path + [action], newPriority), newPriority)

    # return list of None when astar search fails
    return [None]


def pathToPosition(pos: tuple, path: list) -> tuple:
    """From start point and a path, resolve the destination.

    Arguments:
    - pos -- Start point
    - path -- List of actions

    Return:
    - Destination point
    """
    for action in path:
        dx, dy = game.Actions.directionToVector(action)
        pos = (pos[0] + dx, pos[1] + dy)

    return pos


def stepsToPosition(pos: tuple, path: list, n: int) -> tuple:
    """Same as pathToPosition, but only take n steps.

    Arguments:
    - pos -- Start point
    - path -- List of actions
    - n -- Number of steps to take

    Return:
    - Point reached after n steps
    """
    if n >= len(path):
        return pathToPosition(pos, path)

    for i in range(n):
        action = path[i]
        dx, dy = game.Actions.directionToVector(action)
        pos = (pos[0] + dx, pos[1] + dy)

    return pos


"""Agents

Agents are classes which acts in a certain manner on every turn.
Methods of Agents are to decide what kind of actions (e.g. farming,
runaway, hunt...) to take. After this decision has been done,
corresponding Actor class calculates actual action (e.g. NORTH, SOUTH, ...)

Agents are able to switch its operation mode by replacing Actors.
"""

class OffensiveAgent(CaptureAgent):
    """
    OffensiveAgent infiltrates into enemy terrirory, and begin farming
    for food and power pellets. This agent uses astar search for default,
    but switch to minimax mode when enemy ghosts are nearby.
    """

    def __init__(self, index):
        super().__init__(index)


    def registerInitialState(self, gameState):
        """Initialize an agent at the first turn.
        Acts like a constructor.
        """
        self.actor = OffensiveActor(self)
        self.target = None          # food position to eat
        self.ateFood = 0            # number of carrying food
        self.objective = 0          # number of food wish to gather
        self.stopCount = 0          # number of detected consecutive stops

        # parent constructor
        CaptureAgent.registerInitialState(self, gameState)
        
        width = gameState.data.layout.width
        if self.red:
            self.homeRange = range(0, width // 2)
        else:
            self.homeRange = range(width // 2, width)

        self.objective = self.updateObjective(gameState)
    

    def updateObjective(self, gameState):
        """Rearrange objective as 25 percent of food on map.
        """
        food = self.getFood(gameState).asList()
        return len(food) // 4


    def inHome(self, gameState) -> bool:
        """Return true when the agent is in the home area.
        """
        pos = gameState.getAgentPosition(self.index)
        width = gameState.data.layout.width
        if self.red:
            return pos[0] < width // 2
        else:
            return pos[0] >= width // 2

    
    def getGhosts(self, gameState) -> list:
        """Return a list of ghosts in the enemy territory.
        """
        enemies = [
            gameState.getAgentState(i)
            for i in self.getOpponents(gameState)
        ]
        ghosts = [
            e.getPosition() for e in enemies
            if not e.isPacman and e.getPosition() != None and 
            e.scaredTimer < 1
        ]
        
        return ghosts

    
    def getWeakenedGhosts(self, gameState) -> list:
        """Return a list of weakened ghosts in the enemy territory.
        """
        enemies = [
            gameState.getAgentState(i)
            for i in self.getOpponents(gameState)
        ]
        weakendGhosts = [
            e.getPosition() for e in enemies
            if not e.isPacman and e.getPosition() != None and 
            e.scaredTimer < 4 and e.scaredTimer >= 1
        ]
        
        return weakendGhosts


    def capsuleActive(self, gameState) -> bool:
        """Return true when there are still weakened ghosts.
        This means that the effect of power pellet is still active.
        """
        return len(self.getWeakenedGhosts(gameState)) > 0

        
    def areGhostsNearby(self, gameState) -> bool:
        """Return true when ghosts are reachable in 3 steps.
        """
        ghosts = self.getGhosts(gameState)
        pos = gameState.getAgentPosition(self.index)
        for ghPos in ghosts:
            if self.getMazeDistance(pos, ghPos) < 4:
                return True

        return False


    def isRecallPossible(self, gameState) -> bool:
        """Return true when going back to home without being captured
        is possible.
        """
        ghosts = self.getGhosts(gameState)
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.index)
        path = bfsEvade(pos, walls, lambda x, y: x in self.homeRange, ghosts)

        return path[0] != None


    def isSafeFood(self, foodPos, foodLen, gameState) -> bool:
        """[EXPERIMENTAL] Return true when the food is not dead-end.
        """
        ghosts = self.getGhosts(gameState)
        pos = gameState.getAgentPosition(self.index)
        walls = gameState.getWalls()

        nextGhosts = []
        for ghPos in ghosts:
            ghPath = bfs(ghPos, walls, lambda x, y: (x, y) == foodPos)
            nextGhosts.append(stepsToPosition(ghPos, ghPath, foodLen))

        path = bfsEvade(foodPos, walls, lambda x, y: x in self.homeRange, nextGhosts)

        return path[0] != None


    def astarSafe(self, pos: tuple, walls: list, goalTest, heuristic, gameState) -> list:
        """[EXPERIMENTAL] astar search which seeks for not dead-end food.
        """
        pq = util.PriorityQueue()
        current = (pos, [], 0)
        pq.push(current, 0)
        visited = set()

        while not pq.isEmpty():
            node, path, priority = pq.pop()
            if goalTest(node[0], node[1]) and self.isSafeFood(node, len(path), gameState):
                return path
            if not node in visited:
                visited.add(node)
                neighbors = game.Actions.getLegalNeighbors(node, walls)
                for nx, ny in neighbors:
                    dx, dy = nx - node[0], ny - node[1]
                    action = game.Actions.vectorToDirection((dx, dy))
                    newPriority = priority + 1 + heuristic((nx, ny), gameState)
                    pq.push(((nx, ny), path + [action], newPriority), newPriority)

        # return list of None when astar search fails
        return [None]

    
    def isCapsuleFavorable(self, gameState) -> bool:
        """Return true when power pellet is in 11 steps and
        eating the pellet without being eaten is possible.
        """
        ghosts = self.getGhosts(gameState)
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.index)
        capsules = self.getCapsules(gameState)
        path = bfsEvade(pos, walls, lambda x, y: (x, y) in capsules, ghosts)

        return path[0] != None and len(path) < 12


    def isHuntingFavorable(self, gameState) -> bool:
        """Return true when weakened ghosts are in 5 steps.
        """
        ghosts, weakendGhosts = self.getGhosts(gameState), self.getWeakenedGhosts(gameState)
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.index)
        path = bfsEvade(pos, walls, lambda x, y: (x, y) in weakendGhosts, ghosts)

        return path[0] != None and len(path) < 6


    def riskHeuristic(self, pos: tuple, gameState) -> float:
        """Heuristic functions for food search. Gives penalty for food
        close to the ghosts.
        """
        ghosts = self.getGhosts(gameState)

        risk = 0
        for ghPos in ghosts:
            risk += 100.0 / pow(self.getMazeDistance(pos, ghPos) + 1, 2)

        return risk


    def getFoodTarget(self, gameState) -> tuple:
        """Return position of most favorable food to eat.
        """
        food = self.getFood(gameState).asList()
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.index)

        path = astar(
            pos, walls, lambda x, y: (x, y) in food, self.riskHeuristic, gameState
        )

        return pathToPosition(pos, path)


    def getSafeFoodTarget(self, gameState) -> tuple:
        """[EXPERIMENTAL] Return position of most favorable food to eat
        which is not a dead-end.
        """
        food = self.getFood(gameState).asList()
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.index)

        path = self.astarSafe(
            pos, walls, lambda x, y: (x, y) in food, self.riskHeuristic, gameState
        )
        if path[0] == None:
            return (None, None)

        return pathToPosition(pos, path)


    def getCapsuleTarget(self, gameState) -> tuple:
        """Return position of most favorable power pellet to eat.
        """
        capsules = self.getCapsules(gameState)
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.index)

        path = astar(
            pos, walls, lambda x, y: (x, y) in capsules, self.riskHeuristic, gameState
        )
        
        return pathToPosition(pos, path)


    def chooseAction(self, gameState) -> str:
        """Choose action to take on this turn. This method is called
        every turn on every agents.
        """
        # when the agent ate enough food
        if self.ateFood >= self.objective:

            # recall when possible
            if self.isRecallPossible(gameState):
                action = self.actor.gotoHome(gameState)
                return self.confirm(action, gameState)

            else:   # otherwise kill itself
                action = self.actor.goSuicide(gameState)
                return self.confirm(action, gameState)       

        # when ghosts are nearby
        if self.areGhostsNearby(gameState):

            # recall when possible
            if self.isRecallPossible(gameState):
                action = self.actor.gotoHome(gameState)
                return self.confirm(action, gameState)

            # otherwise, switch to minimax mode
            # head for power pellet when favorable, otherwise head for food
            if self.isCapsuleFavorable(gameState) and not self.capsuleActive(gameState):
                target = self.getCapsuleTarget(gameState)
            else:
                target = self.getSafeFoodTarget(gameState)
                if target == (None, None):
                    target = self.getFoodTarget(gameState)

            actor2 = MinimaxActor(self, target, gameState)
            action = actor2.getAction(gameState)
            return self.confirm(action, gameState)
            
        # head for power pellet when favorable
        if self.isCapsuleFavorable(gameState) and not self.capsuleActive(gameState):
            action = self.actor.gotoCapsule(gameState)
            return self.confirm(action, gameState)

        # hunt weakened ghosts when favorable
        # there is no bonus for hunting, but this eases farming difficulty
        if self.isHuntingFavorable(gameState):
            action = self.actor.gotoHunt(gameState)
            return self.confirm(action, gameState)

        # when in homeland, rush to the food to eat
        if self.inHome(gameState):
            action = self.actor.gotoSafeFood(gameState)
            return self.confirm(action, gameState)

        # otherwise seek for foods in default
        action = self.actor.gotoSafeFood(gameState)

        return self.confirm(action, gameState)


    def confirm(self, action, gameState) -> str:
        """Inspect the chosen action. This function is always called
        after OffensiveAgent decides its action.
        """
        # when action is None, this means that the search had failed
        # for various reasons. agent should stop in this turn
        if action == None:
            action = 'Stop'

        # action is normal. inspect the action
        else:
            ghosts = self.getGhosts(gameState)
            food = self.getFood(gameState).asList()
            pos = gameState.getAgentPosition(self.index)

            dx, dy = game.Actions.directionToVector(action)
            nextPos = (pos[0] + dx, pos[1] + dy)

            # agent eats food in next turn without being eaten
            if nextPos not in ghosts and nextPos in food:
                self.ateFood += 1

            # agent is in home after recall or death
            # rearrange the objective depending number of foods on the map
            if pos in self.homeRange or nextPos[0] in self.homeRange:
                self.ateFood = 0
                self.objective = self.updateObjective(gameState)

        # when over 4 consecutive stops were detected, something gone wrong
        # switch to random mode in this turn
        if action == 'Stop':
            self.stopCount += 1
        else:
            self.stopCount = 0

        if self.stopCount > 3:
            action = self.actor.gotoRandom(gameState)
            self.stopCount = 0

        return action


class DefensiveAgent(CaptureAgent):
    """
    DefensiveAgent defends its homeland from enemy pacmans.
    The agent holds its position defending food or power pellet
    and chase the enemies when there are incoming infiltrators.
    """

    def __init__(self, index):
        super().__init__(index)


    def registerInitialState(self, gameState):
        """Initialize an agent at the first turn.
        Acts like a constructor.
        """
        self.actor = DefensiveActor(self)
        self.target = None          # position to head

        width = gameState.data.layout.width
        if self.red:
            self.homeRange = range(0, width // 2)
        else:
            self.homeRange = range(width // 2, width)

        # parent constructor
        CaptureAgent.registerInitialState(self, gameState)


    def getInvaders(self, gameState) -> list:
        """Return list of infiltrators in homeland
        """
        enemies = [
            gameState.getAgentState(i)
            for i in self.getOpponents(gameState)
        ]
        invaders = [
            e.getPosition() for e in enemies
            if e.isPacman and e.getPosition() != None
        ]

        return invaders


    def getTarget(self, gameState) -> tuple:
        """Return food position which is closest to enemies.
        """
        enemies = [
            gameState.getAgentState(i)
            for i in self.getOpponents(gameState)
        ]
        enemyPositions = [
            e.getPosition() for e in enemies
            if e.getPosition() != None
        ]
        defending = self.getFoodYouAreDefending(gameState).asList()
        walls = gameState.getWalls()

        target, minPath = None, [None] * 65535
        for enPos in enemyPositions:
            enPath = bfs(enPos, walls, lambda x, y: (x, y) in defending)
            if len(enPath) < len(minPath):
                minPath = enPath
                target = pathToPosition(enPos, enPath)

        return target


    def isScared(self, gameState):
        """Return true when the agent is scared.
        This means that enemy has eaten the power pellet in homeland.
        """
        return gameState.getAgentState(self.index).scaredTimer > 0


    def chooseAction(self, gameState):
        """Choose action to take on this turn. This method is called
        every turn on every agents.
        """
        # when enemy has eaten the power pellet,
        # switch to minimax mode to avoid reinforced infiltrators
        if self.isScared(gameState):
            self.target = self.getTarget(gameState)
            actor2 = MinimaxDefenseActor(self, self.target, gameState)
            action = actor2.getAction(gameState)
            return self.confirm(action, gameState)
        
        # when there are infiltrators in homeland, chase them
        invaders = self.getInvaders(gameState)
        if invaders:
            action = self.actor.gotoInvader(gameState, invaders)
            return DefensiveAgent.confirm(self, action, gameState)

        # otherwise, go to food in homeland which
        # enemies would like to prefer most
        self.target = self.getTarget(gameState)
        action = self.actor.gotoTarget(gameState, self.target)

        return DefensiveAgent.confirm(self, action, gameState)

    
    def confirm(self, action, gameState):
        """Inspect the chosen action. This function is always called
        after DefensiveAgent decides its action.
        """
        # when action is None, this means that the search had failed
        # for various reasons. agent should stop in this turn
        if action == None:
            action = 'Stop'

        return action


"""Actors

Actor instances are member of Agents. Actors calculate the actual
actions (e.g. NORTH, SOUTH, ...) to take using searches. Though there are
cases where some redundant computations happens both in Agents and Actors,
we leave them for simplicity since the overhead does not exceed 
the time limit.
"""

class OffensiveActor:
    """Default Actor class for OffensiveActor
    """

    def __init__(self, agent):
        self.agent = agent      # agent index who uses this actor


    def gotoFood(self, gameState):
        food = self.agent.getFood(gameState).asList()
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.agent.index)

        action = astar(
            pos, walls, lambda x, y: (x, y) in food, self.agent.riskHeuristic, gameState
        )[0]

        return action


    def gotoSafeFood(self, gameState):
        food = self.agent.getFood(gameState).asList()
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.agent.index)

        action = self.agent.astarSafe(
            pos, walls, lambda x, y: (x, y) in food, self.agent.riskHeuristic, gameState
        )[0]

        return action

    
    def gotoRandom(self, gameState):
        ghosts = self.agent.getGhosts(gameState)
        pos = gameState.getAgentPosition(self.agent.index)
        legalMoves = gameState.getLegalActions(self.agent.index)

        randomActions = []
        for action in legalMoves:
            dx, dy = game.Actions.directionToVector(action)
            newPos = (pos[0] + dx, pos[1] + dy)
            if newPos not in ghosts:
                randomActions.append(action)

        return random.choice(randomActions)


    def gotoCapsule(self, gameState):
        ghosts = self.agent.getGhosts(gameState)
        capsules = self.agent.getCapsules(gameState)
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.agent.index)

        return bfsEvade(pos, walls, lambda x, y: (x, y) in capsules, ghosts)[0]


    def gotoHunt(self, gameState):
        ghosts, weakendGhosts = (
            self.agent.getGhosts(gameState), self.agent.getWeakenedGhosts(gameState)
        )
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.agent.index)

        return bfsEvade(pos, walls, lambda x, y: (x, y) in weakendGhosts, ghosts)[0]


    def gotoHome(self, gameState):
        ghosts = self.agent.getGhosts(gameState)
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.agent.index)

        return bfsEvade(pos, walls, lambda x, y: x in self.agent.homeRange, ghosts)[0]


    def goSuicide(self, gameState):
        ghosts = self.agent.getGhosts(gameState)
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.agent.index)

        return bfs(pos, walls, lambda x, y: (x, y) in ghosts)[0]


class DefensiveActor():
    """Default Actor class for DefensiveActor
    """

    def __init__(self, agent):
        self.agent = agent      # agent index who uses this actor


    def gotoInvader(self, gameState, invaders):
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.agent.index)

        return bfs(pos, walls, lambda x, y: (x, y) in invaders)[0]

    
    def gotoCapsule(self, gameState, capsules):
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.agent.index)

        path = bfs(pos, walls, lambda x, y: (x, y) in capsules)
        self.agent.target = pathToPosition(pos, path)

        return path[0]


    def gotoTarget(self, gameState, target):
        walls = gameState.getWalls()
        pos = gameState.getAgentPosition(self.agent.index)

        return bfs(pos, walls, lambda x, y: (x, y) == target)[0]


class MinimaxActor:
    """Base actor for minimax mode agents. This actor
    is used when OffensiveAgent switches to minimax mode.
    """
    
    def __init__(self, agent, target, gameState):
        self.agent = agent      # agent index who uses this actor
        self.target = target    # position wish to reach
        self.depth = 2          # search depth for minimax
        self.cyclic = []        # list of agent and enemy ghosts

        self.enemies = [
            (i, gameState.getAgentState(i))
            for i in self.agent.getOpponents(gameState)
        ]
        self.ghosts = [
            (i, e.getPosition()) for i, e in self.enemies
            if not e.isPacman and e.getPosition() != None
            and e.scaredTimer == 0
        ]
        self.cyclic.append(self.agent.index)
        for g in self.ghosts:
            self.cyclic.append(g[0])


    def getCyclicIdx(self, agentId):
        for i in range(len(self.cyclic)):
            if self.cyclic[i] == agentId:
                return i
        return None

    
    def evaluate(self, gameState):
        pos = gameState.getAgentPosition(self.agent.index)
        score = 100.0

        # make the agent head for the target
        score -= 1.0 * self.agent.getMazeDistance(pos, self.target)

        # score is decremented severe when being eaten
        for g in self.ghosts:
            if util.manhattanDistance(pos, g[1]) <= 1:
                score -= 100.0

        return score

    
    def minValue(self, gameState, agentId, depth):
        score = math.inf
        legalMoves = gameState.getLegalActions(agentId)
        if 'Stop' in legalMoves:
            legalMoves.remove('Stop')
        
        cyclicIdx = self.getCyclicIdx(agentId)
        if cyclicIdx == len(self.cyclic) - 1:
            depth += 1
        nextAgent = self.cyclic[(cyclicIdx + 1) % len(self.cyclic)]

        bestAction = None
        for action in legalMoves:
            succState = gameState.generateSuccessor(agentId, action)
            newScore = self.getScoreWithAction(succState, nextAgent, depth)[0]
            if newScore <= score:
                score, bestAction = newScore, action

        return score, bestAction

    
    def maxValue(self, gameState, agentId, depth):
        score = -math.inf
        legalMoves = gameState.getLegalActions(agentId)
        if 'Stop' in legalMoves:
            legalMoves.remove('Stop')
        
        cyclicIdx = self.getCyclicIdx(agentId)
        if cyclicIdx == len(self.cyclic) - 1:
            depth += 1
        nextAgent = self.cyclic[(cyclicIdx + 1) % len(self.cyclic)]

        bestAction = None
        for action in legalMoves:
            succState = gameState.generateSuccessor(agentId, action)
            newScore = self.getScoreWithAction(succState, nextAgent, depth)[0]
            if newScore >= score:
                score, bestAction = newScore, action

        return score, bestAction


    def getScoreWithAction(self, gameState, agentId, depth):
        if depth == self.depth:
            return self.evaluate(gameState), None
        elif self.getCyclicIdx(agentId) == 0:
            return self.maxValue(gameState, agentId, depth)
        else:
            return self.minValue(gameState, agentId, depth)


    def getAction(self, gameState):
        bestAction = self.getScoreWithAction(gameState, self.agent.index, 0)[1]
        
        return bestAction


class MinimaxDefenseActor(MinimaxActor):
    """Minimax Actor for DefensiveAgent
    """
    
    def __init__(self, agent, target, gameState):
        self.agent = agent
        self.target = target
        self.depth = 2
        self.cyclic = []

        self.enemies = [
            (i, gameState.getAgentState(i))
            for i in self.agent.getOpponents(gameState)
        ]
        self.ghosts = [
            (i, e.getPosition()) for i, e in self.enemies
            if e.isPacman and e.getPosition() != None
        ]
        self.cyclic.append(self.agent.index)
        for g in self.ghosts:
            self.cyclic.append(g[0])