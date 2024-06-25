from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import numpy as np
import os.path

start = False
switch = True
count = 0

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts (is an array of all the distances)
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # closest ghost (an index of the closest ghost)
        #smallestDistance = min(gameState.data.ghostDistances)
        #closestGhostIndex = gameState.data.ghostDistances.index(smallestDistance)
        #print("Closest Ghost Index: ", closestGhostIndex)
        # XDirection of pacman to closest ghost (right, left, neither)
            # pacManX - ghostX = left or right based on sign
      #  allGhostPositions = gameState.getGhostPositions()
        #ghostX = allGhostPositions[closestGhostIndex].xcor()
        #pacX = gameState.getPacmanPosition().xcor()
        #Xatr = pacX - ghostX
        #XDir
        #if Xatr > 0 :
        #    XDir = 1 # 1 means left
        #elif Xatr < 0 : 
        #    XDir = -1 # -1 means right
        #else :
        #    XDir = 0 # 0 means that they are equal
        #print("XDirection: ", XDir) 
        # YDirection of pacman to cloest ghost (up, down, neither)
            # pacManY - ghostY = up or down based on sign
        #ghostY = allGhostPositions[closestGhostIndex].ycor()
        #pacY = gameState.getPacmanPosition().ycor()
        #Yatr = pacY - ghostY
        #if Yatr > 0 :
        #    YDir = 1 # 1 means down
        #elif Yatr < 0 : 
        #    YDir = -1 # -1 means up
        #else :
        #    YDir = 0 # 0 means that they are equal
        #print("YDirection: ", YDir)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printLineData(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        count = 0
        global switch
        global start
        if start == False:
            if Directions.WEST in legal: move = Directions.WEST
            elif Directions.NORTH in legal: move = Directions.NORTH
            else: start = True
        if start == True and switch == True:
            if  Directions.EAST in legal: move = Directions.EAST
            elif Directions.SOUTH in legal: 
                move = Directions.SOUTH
                switch = False
        elif start == True and switch == False:
            if Directions.WEST in legal: move = Directions.WEST
            elif Directions.SOUTH in legal:
                move = Directions.SOUTH
                switch = True
            else: switch = True
        #if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

    def printLineData(self, gameState):
        return "XXXXXXXXXX"


class QLearningAgent(BustersAgent):

    #Initialization
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.epsilon = 0.0
        self.alpha = 0.0
        self.discount = 0.8
        self.actions = {"North":0, "East":1, "South":2, "West":3}
        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            #"*** CHECK: NUMBER OF ROWS IN QTABLE DEPENDS ON THE NUMBER OF STATES ***"
            self.initializeQtable(9)

    def initializeQtable(self, nrows):
        "Initialize qtable"
        self.q_table = np.zeros((nrows,len(self.actions)))

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)
           
        return q_table


    def writeQtable(self):
        "Write qtable to disc"        
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")   
            

    def __del__(self):
        "Destructor. Invokation at the end of each episode"        
        self.writeQtable()
        self.table_file.close()

   
    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        """
        "*** YOUR CODE HERE ***"   
        #  util.raiseNotDefined() 
        # my attributes are: 
        # XDirection pacManX - ghostX = left or right based on sign
        # YDirection pacManY - ghostY = up or down based on sign

        # closest ghost (an index of the closest ghost)
        smallestDistance = min(state.data.ghostDistances)
        closestGhostIndex = state.data.ghostDistances.index(smallestDistance)
        #print("Closest Ghost Index: ", closestGhostIndex)
        # XDirection of pacman to closest ghost (right, left, neither)
            # pacManX - ghostX = left or right based on sign
        allGhostPositions = state.getGhostPositions()
        ghostX = allGhostPositions[closestGhostIndex]
        ghostX = ghostX[0]
        #print(ghostX)
        pacX = state.getPacmanPosition()[0]
        #print(pacX)
        Xatr = pacX - ghostX
        #print(Xatr)
        XDir = 0
        if Xatr > 0 :
            XDir = 1 # 1 means left
        elif Xatr < 0 : 
            XDir = -1 # -1 means right
        else :
            XDir = 0 # 0 means that they are equal
        #print("XDirection: ", XDir) 
        # YDirection of pacman to cloest ghost (up, down, neither)
        # pacManY - ghostY = up or down based on sign
        ghostY = allGhostPositions[closestGhostIndex]
        ghostY = ghostY[1]
        pacY = state.getPacmanPosition()[1]
        Yatr = pacY - ghostY
        if Yatr > 0 :
            YDir = 1 # 1 means down
        elif Yatr < 0 : 
            YDir = -1 # -1 means up
        else :
            YDir = 0 # 0 means that they are equal
        #print("YDirection: ", YDir)
        return (XDir, YDir)

    def getQValue(self, state, action):

        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        #position = self.computePosition(state)
        #action_column = self.actions[action]
        #return self.q_table[position][action_column]
        if (state,action) in self.q_table:
            return self.q_table[(state,action)]
        else:
            return 0.0


    def computeValueFromQValues(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.  Note that if
            there are no legal actions, which is the case at the
            terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions)==0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions)==0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        action = None

        if len(legalActions) == 0:
                return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)


    def update(self, state, action, nextState, reward):
        """r = reward
        position = self.computePosition(state)
        action_column = self.actions[action]
        # terminal state if # of legal actions = 0
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0: #in terminal state
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(state, action) + self.alpha * (r + 0) # assigning Q(state,action)
        else: #not in terminal state
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(state, action) + self.alpha * (r + 0) # assigning Q(state,action) 

            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))
        
        """

        "*** YOUR CODE HERE ***"     
        r = reward
        position = self.computePosition(state)
        action_column = self.actions[action]
        # terminal state if # of legal actions = 0
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0: #in terminal state
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(state, action) + self.alpha * (r + 0) # assigning Q(state,action)
        else: #not in terminal state
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(state, action) + self.alpha * (r + 0) # assigning Q(state,action) 

        #util.raiseNotDefined()



    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"        
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"        
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextstate):
        "Return the obtained reward"
        
        "*** YOUR CODE HERE ***" 
        #util.raiseNotDefined()

        dirX = self.computeAction(state).xcor()
        dirY = self.computeAction(state).ycor()
        # dirX tells you what x dir (left 1, right -1, neither 0) the pacman should go to reach the closest ghost
        # dirY tells you what y dir (up -1,down 1, neither 0) the pacman should go to reach the closest ghost
        # the reward should be given if the pacman's action follows what it should do to reach the closest ghost

        #action can be Directions.NORTH etc for all directions
        #reward = -1 if you go the opposite direction of what you are supposed to do.
        #reward = -0.5 if you aren't supposed to go a direction and you go a direction
        #reward = 1 if you go one of the correct directions of what you are supposed to do.

        #QUESTION: should moving in the opposite direction & moving in a direction when you shouldnt move be punished the same?

        #moves in the Y cordinate
        if(action == Directions.NORTH): #pacman moves up
            if(dirY == -1): #means that the pacman should go up to reach the closest ghost
                return 1
            elif (dirY == 1): #means that the pacman should go down to reach the closest ghost
                return -1
            else: #means that the pacman should go neither up or down to reach the closest ghost
                return -0.5
        if(action == Directions.SOUTH): #pacman moves down
            if(dirY == -1): #means that the pacman should go up to reach the closest ghost
                return -1
            elif (dirY == 1): #means that the pacman should go down to reach the closest ghost
                return 1
            else: #means that the pacman should go neither up or down to reach the closest ghost
                return -0.5
        
        #moves in the X cordinate
        if(action == Directions.EAST): #pacman moves right
            if(dirY == -1): #means that the pacman should go right to reach the closest ghost
                return 1
            elif (dirY == 1): #means that the pacman should go left to reach the closest ghost
                return -1
            else: #means that the pacman should go neither left or right to reach the closest ghost
                return -0.5
        if(action == Directions.WEST): #pacman moves left
            if(dirY == -1): #means that the pacman should go right to reach the closest ghost
                return -1
            elif (dirY == 1): #means that the pacman should go left to reach the closest ghost
                return 1
            else: #means that the pacman should go neither right or left to reach the closest ghost
                return -0.5
        



