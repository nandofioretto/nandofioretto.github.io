# myAgents.py
# ---------------
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

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""

targetList = []
numPac = 0
def createAgents(num_pacmen, agent='MyAgent'):
    global numPac
    numPac = num_pacmen
    return [eval(agent)(index=i) for i in range(num_pacmen)]



class MyAgent(Agent):
    """
    Implementation of your agent.
    """
    foodPath = []

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        if len(self.foodPath) == 0:
            self.getActionPath(state)
            
        action = self.foodPath[0]
        self.foodPath.pop(0)
        return action

    
    def getActionPath(self,state):
        #startPosition = state.getPacmanPosition(self.index)
        #food = state.getFood()
        #walls = state.getWalls()
        problem = AnyFoodSearchProblem(state,self.index)
        #foodList = food.asList()
        #global targetList

        self.foodPath = search.bfs(problem)
        #self.foodPath = search.ucs(problem)
        
        
        # closeDot = (-1,-1)
        # closeDis = 99999
        # for F in foodList:
        #     dis = util.manhattanDistance(startPosition, F)
        #     if dis < closeDis:
        #         closeDot = F
        # prob = PositionSearchProblem(state, start=startPosition, goal=closeDot, warn=False, visualize=False)
        # self.foodPath = search.bfs(prob)


    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        # print('HEEELELELLEKJDOISDHJFOISDEJOLKFJSDLFJ')
        # global targetList
        # targetList.clear()

        #raise NotImplementedError()

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):
    foodPath = []

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState,self.index)
        #foodList = food.asList()
        #print("foodList",foodList)
        #nearestFood = (0,0)
        #nearestSteps = 99999

        
        if len(self.foodPath) == 0:
            self.foodPath = search.bfs(problem)
            # for foodPosition in foodList:
            #     path = search.bfs(problem)
            #     if len(path) < nearestSteps: 
            #         foodPath = path
            #         nearestSteps = len(path)
        if len(self.foodPath) > 0:
            action = self.foodPath[0]
            self.foodPath.pop(0)
            return action
        
        return 'Stop'
        #prob = PositionSearchProblem(gameState, start=startPosition, goal=foodPosition, warn=False, visualize=False)
        #return search.bfs(prob)
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

    def getAction(self, state):
        if len(self.foodPath) > 0:
            action = self.foodPath[0]
            self.foodPath.pop(0)
            return action
        else:
            return self.findPathToClosestDot(state)
        #return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """
    

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()
        self.foodList = self.food.asList()
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """

        x,y = state
        global targetList,numPac
        #if(len(targetList) > 10): targetList.clear()
        "*** YOUR CODE HERE ***"
        if self.food[x][y] == 1:
            if len(self.foodList) <= numPac:
                return True
            if (x,y) not in targetList:
                targetList.append((x,y))
                if len(targetList) > 10:
                    targetList.pop(0)
                return True
        return False
        

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])