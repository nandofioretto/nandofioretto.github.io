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
import functools
from search import breadthFirstSearch
from game import Directions

# store a global scheduler
# which is shared between agents
agent_scheduler = None

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    global agent_scheduler
    # initialize the global variable to None
    # so that information will get recomputed for each map
    agent_scheduler = None
    return [eval(agent)(index=i) for i in range(num_pacmen)]

class MapBlock:
    """
    Helper class for defining each splitted map block
    """
    def __init__(self, assignedID, foods, center):
        self.ID = assignedID # each block has an ID
        self.foods = foods # food positions as list in this area
        self.center = center # approaximated center

class MapBlockScheduler:
    """
    Helper class for assigning an agent a map area to search
    """
    def __init__(self, cutsetsWeighted, mapInfo, agentLocations):
        self.cutsets = cutsetsWeighted
        self.mapInfo = mapInfo
        self.agentsPos = agentLocations
        self.visited = []
        self._init_agent_states()

    def getFoods(self, agentId):
        """
        Query the agent's current position in map block id\\
        Then returns the nearest map block foods based on the cutsets
        """
        if len(self.mapInfo) <= 0: return []
        mapid = self._graph_nearest(agentId)
        if mapid < 0: return []
        self.visited.append(mapid)
        self.agentsPos[agentId] = mapid
        return self.mapInfo[mapid].foods

    def updateFoods(self, agentId, numFoods, done):
        """
        Update agent status
        """
        self.agentStates[agentId] = (done, numFoods)

    def _init_agent_states(self):
        """
        Initialize agent status
        """
        self.agentStates = {}
        for agentId in self.agentsPos.keys():
            self.agentStates[agentId] = (False, 0) # (done, remaining food)

    def _graph_nearest(self, agentId):
        """
        Return the nearest map block id from query agent\\
        May return -1, if there's no better map choice
        """
        from util import Queue
        # first find the nearest map block
        queue = Queue()
        queue.push((self.agentsPos[agentId], 0))
        nearestId = None
        nearestPathCost = 0xFFFFFFFF
        visited = []
        while not queue.isEmpty():
            mapid, pathcost = queue.pop()
            if mapid not in visited:
                visited.append(mapid)
                if mapid in self.visited:
                    [queue.push((connID, pathcost+connCost)) for connID, connCost in self.cutsets[mapid] if connID not in visited]
                elif pathcost < nearestPathCost:
                    nearestPathCost = pathcost
                    nearestId = mapid
        if nearestId is None: return -1
        # next search any nearby agent that is much nearer than this agent
        nearbyAgentId = None
        visited = []
        queue = Queue()
        queue.push((nearestId, 0, 1)) # (mapid, cost, depth)
        while not queue.isEmpty():
            mapid, pathcost, depth = queue.pop()
            if (depth >= 3) or (pathcost > nearestPathCost): continue
            if mapid not in visited:
                visited.append(mapid)
                for aId, mId in self.agentsPos.items():
                    if self.agentStates[aId][0]: continue
                    if mId == mapid:
                        approxcost = pathcost + 20*(self.agentStates[aId][1])
                        if approxcost < nearestPathCost:
                            nearbyAgentId = aId
                        break
                if nearbyAgentId is not None: break
                [queue.push((connID, pathcost+connCost+20*len(self.mapInfo[connID].foods), depth+1)) for connID, connCost in self.cutsets[mapid] if connID not in visited]
        
        if nearbyAgentId is not None: return -1
        # finally return the mapid
        return nearestId
        

class MyAgent(Agent):
    """
    Implementation of your agent.
    """

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        global agent_scheduler

        selfPos = state.getPacmanPosition(self.index)
        walls = state.getWalls()
        foods = state.getFood()

        if (self.index == 0) and (agent_scheduler is None):
            agent_scheduler = self.processMap(state)

        if self.done: return Directions.STOP

        if (self.current_target is not None) and (not foods[self.current_target[0]][self.current_target[1]]):
            self.current_target = None
            self.current_actions = []

        foodsAlreadyVisited = []
        [foodsAlreadyVisited.append(foodPos) for foodPos in self.area_foods if not foods[foodPos[0]][foodPos[1]]]
        [self.area_foods.remove(foodPos) for foodPos in foodsAlreadyVisited]

        if len(self.current_actions) > 0:
            nextAction = self.current_actions[0]
            self.current_actions = self.current_actions[1:]
            return nextAction

        if len(self.area_foods) <= 0:
            self.area_foods = agent_scheduler.getFoods(self.index)
            if len(self.area_foods) <= 0:
                self.done = True
                agent_scheduler.updateFoods(self.index, 0, self.done)
                return Directions.STOP
        
        self.current_target = self.findNearestFood(selfPos, self.area_foods, walls)
        self.current_actions = breadthFirstSearch(PositionSearchProblem(state, start=selfPos, goal=self.current_target, warn=False, visualize=False))
        nextAction = self.current_actions[0]
        self.current_actions = self.current_actions[1:]
        agent_scheduler.updateFoods(self.index, len(self.area_foods), self.done)
        return nextAction

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        self.current_target = None
        self.current_actions = []
        self.area_foods = []
        self.done = False

    def findNearestFood(self, position, foods, walls):
        """
        Find nearest food from a list, with consideration of walls
        """
        assert len(foods) > 0
        nearest_food = foods[0]
        nearest_food_dist = 0xFFFFFFFF
        for foodPos in foods:
            # dist = self.myDistance(position, foodPos, walls)
            dist = abs(position[0]-foodPos[0]) + abs(position[1]-foodPos[1])
            # get nearest dist
            if dist < nearest_food_dist:
                nearest_food = foodPos
                nearest_food_dist = dist
        return nearest_food

    def myDistance(self, startPos, endPos, walls):
        """
        My distance heuristic with the consideration of walls
        """
        wall_w = walls.width
        wall_h = walls.height
        dist = abs(startPos[0] - endPos[0]) + abs(startPos[1] - endPos[1])
        # scan for vertical walls
        if endPos[0] > startPos[0]:
            i_start = startPos[0]
            i_end = endPos[0]
        else:
            i_start = endPos[0]
            i_end = startPos[0]
        newAdd1 = 0
        newAdd2 = 0
        for i in range(i_start + 1, i_end):
            if walls[i][endPos[1]]:
                # wall detected, try to go around
                dist_up = 0
                dist_down = 0
                for m in range(endPos[1]-1, -1, -1):
                    if walls[i][m]:
                        dist_up += 1
                    else:
                        break
                for n in range(endPos[1]+1, wall_h):
                    if walls[i][n]:
                        dist_down += 1
                    else:
                        break
                newAdd1 += 2 * min(dist_up, dist_down)
            if  walls[i][startPos[1]]:
                # wall detected, try to go around
                dist_up = 0
                dist_down = 0
                for m in range(startPos[1]-1, -1, -1):
                    if walls[i][m]:
                        dist_up += 1
                    else:
                        break
                for n in range(startPos[1]+1, wall_h):
                    if walls[i][n]:
                        dist_down += 1
                    else:
                        break
                newAdd2 += 2 * min(dist_up, dist_down)
        # scan for horizontal walls
        if endPos[1] > startPos[1]:
            j_start = startPos[1]
            j_end = endPos[1]
        else:
            j_start = endPos[1]
            j_end = startPos[1]
        newAdd3 = 0
        newAdd4 = 0
        for j in range(j_start + 1, j_end):
            if walls[startPos[0]][j]:
                # wall detected, try to go around
                dist_up = 0
                dist_down = 0
                for m in range(startPos[0]-1, -1, -1):
                    if walls[m][j]:
                        dist_up += 1
                    else:
                        break
                for n in range(startPos[0]+1, wall_w):
                    if walls[n][j]:
                        dist_down += 1
                    else:
                        break
                newAdd3 += 2 * min(dist_up, dist_down)
            if walls[endPos[0]][j]:
                # wall detected, try to go around
                dist_up = 0
                dist_down = 0
                for m in range(endPos[0]-1, -1, -1):
                    if walls[m][j]:
                        dist_up += 1
                    else:
                        break
                for n in range(endPos[0]+1, wall_w):
                    if walls[n][j]:
                        dist_down += 1
                    else:
                        break
                newAdd4 += 2 * min(dist_up, dist_down)
        dist += min(newAdd1 + newAdd3, newAdd2 + newAdd4)
        return dist

    # @functools.lru_cache(256)
    def processMap(self, state):
        """
        Run by agent 0: split the map into several parts (by splitting foods),
        which can be accessed by all agents
        ------
        Idea is that:\\
        Divide the map to several continuous map blocks, each identified by the food positions in that area\\
        Then create a scheduler based on the map information and map cutsets (a tree search problem)
        """
        from util import Queue
        # prepare information
        walls = state.getWalls()
        foods = state.getFood()
        numAgents = state.getNumAgents()
        numFoods = len(foods.asList())
        W, H = walls.width, walls.height
        mapInfo = {}
        map_cutsets = []
        map_split_count = 8

        # first try to split map to several blocks
        id_count = 0
        grid = [[0 for _ in range(H)] for _ in range(W)]
        split_stride = min(W//map_split_count, H//map_split_count)
        for i in range(0, W, split_stride):
            for j in range(0, H, split_stride):
                area_x_max = min(W, i+split_stride)
                area_y_max = min(H, j+split_stride)
                mapqueue = Queue()
                mapqueuepush = mapqueue.push
                mapqueueisEmpty = mapqueue.isEmpty
                mapqueuepop = mapqueue.pop
                center = ((area_x_max+i)//2, (area_y_max+j)//2)
                for option in [(m,n) for m in [-3,-2,-1,0,1,2,3] for n in [-3,-2,-1,0,1,2,3]]:
                    pos = (center[0]+option[0], center[1]+option[1])
                    if (pos[0] < i) or (pos[1] < j) or (pos[0] >= area_x_max) or (pos[1] >= area_y_max):
                        continue
                    if not walls[pos[0]][pos[1]]:
                        mapqueuepush(pos)
                        break
                if mapqueueisEmpty():
                    # should not be empty
                    continue
                mapmemory = set() # a tmp memory of assigned map areas
                food_in_area = [] # record food in this map area
                cut_sets = [] # record cut sets discovered
                cut_setsappend = cut_sets.append
                while not mapqueueisEmpty():
                    pos = mapqueuepop()
                    mapmemory.add(pos)
                    if foods[pos[0]][pos[1]]:
                        food_in_area.append(pos)
                    if grid[pos[0]][pos[1]] <= 0:
                        grid[pos[0]][pos[1]] = id_count + 1
                    # no limit (except 0) for upper left neighbors
                    if (pos[0] > 0) and (not walls[pos[0]-1][pos[1]]) and (grid[pos[0]-1][pos[1]] <= 0):
                        if (pos[0]-1,pos[1]) not in mapqueue.list:
                            mapqueuepush((pos[0]-1,pos[1]))
                    if (pos[1] > 0) and (not walls[pos[0]][pos[1]-1]) and (grid[pos[0]][pos[1]-1] <= 0):
                        if (pos[0],pos[1]-1) not in mapqueue.list:
                            mapqueuepush((pos[0],pos[1]-1))
                    # with strong limit for lower right neighbors
                    if (pos[0] < area_x_max-1) and (not walls[pos[0]+1][pos[1]]) and (grid[pos[0]+1][pos[1]] <= 0):
                        if (pos[0]+1,pos[1]) not in mapqueue.list:
                            mapqueuepush((pos[0]+1,pos[1]))
                    if (pos[1] < area_y_max-1) and (not walls[pos[0]][pos[1]+1]) and (grid[pos[0]][pos[1]+1] <= 0):
                        if (pos[0],pos[1]+1) not in mapqueue.list:
                            mapqueuepush((pos[0],pos[1]+1))
                    # check for map cutsets
                    if (pos[0] > 0) and (grid[pos[0]-1][pos[1]] > 0) and (grid[pos[0]-1][pos[1]] != (id_count+1)):
                        cut_setsappend((grid[pos[0]-1][pos[1]]-1, id_count))
                    if (pos[1] > 0) and (grid[pos[0]][pos[1]-1] > 0) and (grid[pos[0]][pos[1]-1] != (id_count+1)):
                        cut_setsappend((grid[pos[0]][pos[1]-1]-1, id_count))
                    if (pos[0] < area_x_max-1) and (grid[pos[0]+1][pos[1]] > 0) and (grid[pos[0]+1][pos[1]] != (id_count+1)):
                        cut_setsappend((grid[pos[0]+1][pos[1]]-1, id_count))
                    if (pos[1] < area_y_max-1) and (grid[pos[0]][pos[1]+1] > 0) and (grid[pos[0]][pos[1]+1] != (id_count+1)):
                        cut_setsappend((grid[pos[0]][pos[1]+1]-1, id_count))
                if len(food_in_area) <= 0:
                    for pos in mapmemory:
                        grid[pos[0]][pos[1]] = 0 # reset to 0
                    continue # skip if no food in this area, which is not very possible
                map_cutsets += cut_sets
                # compute an approximate center
                food_x_y = [0, 0xFFFFFFFF, 0, 0xFFFFFFFF]
                for x,y in food_in_area:
                    food_x_y[0] = max(x, food_x_y[0])
                    food_x_y[1] = min(x, food_x_y[1])
                    food_x_y[2] = max(y, food_x_y[2])
                    food_x_y[3] = min(y, food_x_y[3])
                center = ((food_x_y[0]+food_x_y[1])//2, (food_x_y[2]+food_x_y[3])//2)
                new_map_block = MapBlock(id_count, food_in_area, center)
                mapInfo[id_count] = new_map_block
                id_count += 1

        # make sure each free space (non-wall) position is assigned a value
        mapqueue = Queue()
        mapqueuepush = mapqueue.push
        mapqueueisEmpty = mapqueue.isEmpty
        mapqueuepop = mapqueue.pop
        map_cutsetsappend = map_cutsets.append
        for i in range(W):
            for j in range(H):
                if grid[i][j] > 0:
                    continue
                # if not a wall and not assigned, then should assign it
                if not walls[i][j]:
                    mapqueuepush((i,j))
        while not mapqueueisEmpty():
            pos = mapqueuepop()
            if grid[pos[0]][pos[1]] > 0: continue
            if grid[pos[0]-1][pos[1]] > 0:
                grid[pos[0]][pos[1]] = grid[pos[0]-1][pos[1]]
                if foods[pos[0]][pos[1]]:
                    mapInfo[grid[pos[0]][pos[1]]-1].foods.append(pos)
            elif grid[pos[0]][pos[1]-1] > 0:
                grid[pos[0]][pos[1]] = grid[pos[0]][pos[1]-1]
                if foods[pos[0]][pos[1]]:
                    mapInfo[grid[pos[0]][pos[1]]-1].foods.append(pos)
            elif grid[pos[0]+1][pos[1]] > 0:
                grid[pos[0]][pos[1]] = grid[pos[0]+1][pos[1]]
                if foods[pos[0]][pos[1]]:
                    mapInfo[grid[pos[0]][pos[1]]-1].foods.append(pos)
            elif grid[pos[0]][pos[1]+1] > 0:
                grid[pos[0]][pos[1]] = grid[pos[0]][pos[1]+1]
                if foods[pos[0]][pos[1]]:
                    mapInfo[grid[pos[0]][pos[1]]-1].foods.append(pos)
            else:
                mapqueuepush(pos) # push back to queue and wait for assignment
            if grid[pos[0]][pos[1]] > 0:
                if (pos[0] > 0) and (grid[pos[0]-1][pos[1]] > 0) and (grid[pos[0]-1][pos[1]] != grid[pos[0]][pos[1]]):
                    map_cutsetsappend((grid[pos[0]-1][pos[1]]-1, grid[pos[0]][pos[1]]-1))
                if (pos[1] > 0) and (grid[pos[0]][pos[1]-1] > 0) and (grid[pos[0]][pos[1]-1] != grid[pos[0]][pos[1]]):
                    map_cutsetsappend((grid[pos[0]][pos[1]-1]-1, grid[pos[0]][pos[1]]-1))
                if (pos[0] < area_x_max-1) and (grid[pos[0]+1][pos[1]] > 0) and (grid[pos[0]+1][pos[1]] != grid[pos[0]][pos[1]]):
                    map_cutsetsappend((grid[pos[0]+1][pos[1]]-1, grid[pos[0]][pos[1]]-1))
                if (pos[1] < area_y_max-1) and (grid[pos[0]][pos[1]+1] > 0) and (grid[pos[0]][pos[1]+1] != grid[pos[0]][pos[1]]):
                    map_cutsetsappend((grid[pos[0]][pos[1]+1]-1, grid[pos[0]][pos[1]]-1))

        # # process collected cutsets and compute approximate distance between them
        map_cutsets = [tuple(sorted(m)) for m in map_cutsets]
        map_cutsets = set(map_cutsets)
        map_cutsets_weighted = {}
        for cutset in map_cutsets:
            # approximate cost to go from one center to another
            pos1 = mapInfo[cutset[0]].center
            pos2 = mapInfo[cutset[1]].center
            # dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] + pos2[1])
            dist = self.myDistance(pos1, pos2, walls)
            if cutset[0] not in map_cutsets_weighted.keys():
                map_cutsets_weighted[cutset[0]] = [(cutset[1], dist)]
            else:
                map_cutsets_weighted[cutset[0]].append((cutset[1], dist))
            if cutset[1] not in map_cutsets_weighted.keys():
                map_cutsets_weighted[cutset[1]] = [(cutset[0], dist)]
            else:
                map_cutsets_weighted[cutset[1]].append((cutset[0], dist))

        # get all agent locations by map id
        agent_locations = {} # map area ids
        for agentId in range(numAgents):
            agentPos = state.getPacmanPosition(agentId)
            agent_locations[agentId] = grid[agentPos[0]][agentPos[1]]-1

        scheduler = MapBlockScheduler(map_cutsets_weighted, mapInfo, agent_locations)

        return scheduler

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return search.bfs(problem)

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]
        

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

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        goal = self.food[x][y]
        return goal