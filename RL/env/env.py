import socket
import json
import numpy as np

from rts.unit.unit import Unit
from rts.unit.unit_action import UnitAction
from rts.unit.unit_type import UnitType
from rts.unit.unit_type_table import UnitTypeTable

from rts.player.player import Player
from rts.player.player_action import PlayerAction

from rts.physical_game_state import PhysicalGameState

from env.board import Board

DEBUG = False
HOST = "127.0.0.1"
PORT = 10000

class Env:
    def __init__(self):
        # Core variables
        self.clientSocket:      socket.socket                       = None
        self.timeBudget:        int                                 = -1
        self.iterationsBudget:  int                                 = -1
        self.currentMessage:    bytes                               = b''
        self.terrainWidth:      int                                 = 0
        self.terrainHeight:     int                                 = 0
        self.terrainMap:        list[int]                           = []
        self.player:            int                                 = 0
        self.players:           list[Player]                        = []
        self.unitTypeTable:     UnitTypeTable                       = UnitTypeTable()
        self.units:             list[Unit]                          = []
        self.neutralUnits:      list[Unit]                          = []
        self.enemyUnits:        list[Unit]                          = []
        self.maxRange = 0
        self.onGoingActions:    list[list[int, UnitAction]]         = []
        self.availableActions:  list[list[Unit, list[UnitAction]]]  = []
        self.waitForInput:      bool                                = False

        # For display
        self.cell_dimension:    int     = 30
        self.board:             Board   = Board()

        # For AI
        self.observation:       list[float] = []
        self.observation_space: np.array    = 0
        self.action_space:      np.array    = 0
        self.reward:            float       = 0
        self.done:              bool        = False

    def start(self):
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False

        while not connected:
            try:
                self.clientSocket.connect((HOST, PORT))
                connected = True
            except:
                continue

        if DEBUG:
            print("Connected")

        self.ack()

        while not self.waitForInput:
            self.receiveMessage()
            self.processMessage()
        self.board.refresh(self.terrainMap, self.units + self.neutralUnits + self.enemyUnits, self.players)

        self.observation_space = np.zeros((self.terrainWidth, self.terrainHeight, (5 + 5 + 3 + 9 + 6)))
        # In reference to the structuration given in Rapport_ADL_2.pdf page 8.

        self.maxRange = 0
        for unitType in self.unitTypeTable.unitTypes:
            if unitType.attackRange > self.maxRange:
                self.maxRange = unitType.attackRange
        # max_range create a square around each unit to determine the relative position of the attacked unit
        self.action_space = np.zeros((self.terrainWidth * self.terrainHeight, 7))
        # In reference to the structuration given in Rapport_ADL_2.pdf page 8.

        self.computeObservation()

        return self.observation
    
    def computeObservation(self):
        self.observation = np.zeros(self.observation_space.shape)

        for unit in self.units + self.neutralUnits + self.enemyUnits:

            if unit.hitpoints == 0:
                self.observation[unit.x][unit.y][0] = 1
            elif unit.hitpoints == 1:
                self.observation[unit.x][unit.y][1] = 1
            elif unit.hitpoints == 2:
                self.observation[unit.x][unit.y][2] = 1
            elif unit.hitpoints == 3:
                self.observation[unit.x][unit.y][3] = 1
            else:
                self.observation[unit.x][unit.y][4] = 1

            if unit.resources == 0:
                self.observation[unit.x][unit.y][5] = 1
            elif unit.resources == 1:
                self.observation[unit.x][unit.y][6] = 1
            elif unit.resources == 2:
                self.observation[unit.x][unit.y][7] = 1
            elif unit.resources == 3:
                self.observation[unit.x][unit.y][8] = 1
            else:
                self.observation[unit.x][unit.y][9] = 1

            if unit.player == -1:
                self.observation[unit.x][unit.y][10] = 1
            elif unit.resources == 0:
                self.observation[unit.x][unit.y][11] = 1
            else:
                self.observation[unit.x][unit.y][12] = 1

            # One is skipped for the None type
            if unit.type.name == 'Resource':
                self.observation[unit.x][unit.y][14] = 1
            elif unit.type.name == 'Base':
                self.observation[unit.x][unit.y][15] = 1
            elif unit.type.name == 'Barracks':
                self.observation[unit.x][unit.y][16] = 1
            elif unit.type.name == 'Worker':
                self.observation[unit.x][unit.y][17] = 1
            elif unit.type.name == 'Light':
                self.observation[unit.x][unit.y][18] = 1
            elif unit.type.name == 'Heavy':
                self.observation[unit.x][unit.y][19] = 1
            elif unit.type.name == 'Ranged':
                self.observation[unit.x][unit.y][20] = 1
            # One is skipped for the Wall type

            # One is skipped for the None action
            for onGoindAction in self.onGoingActions:
                if onGoindAction[0] == unit.ID:
                    action: UnitAction = onGoindAction[1]
                    if action.type == action.TYPE_MOVE:
                        self.observation[unit.x][unit.y][23] = 1
                    elif action.type == action.TYPE_HARVEST:
                        self.observation[unit.x][unit.y][24] = 1
                    elif action.type == action.TYPE_RETURN:
                        self.observation[unit.x][unit.y][25] = 1
                    elif action.type == action.TYPE_PRODUCE:
                        self.observation[unit.x][unit.y][26] = 1
                    elif action.type == action.TYPE_ATTACK_LOCATION:
                        self.observation[unit.x][unit.y][27] = 1

        for x in range(self.terrainWidth):
            for y in range(self.terrainHeight):
                index = x + y * self.terrainWidth
                if self.terrainMap[index] == PhysicalGameState.TERRAIN_WALL:
                    self.observation[x][y][21] = 1
            
    def step(self, action):
        playerAction = PlayerAction()

        if action != None:
            playerAction.addAction(action[0], action[1])

        self.clientSocket.sendall(bytes(playerAction.toJSON(), encoding='utf-8') + b'\n')

        self.waitForInput = False
        while not self.waitForInput:
            self.receiveMessage()
            self.processMessage()
        self.board.refresh(self.terrainMap, self.units + self.neutralUnits + self.enemyUnits, self.players)

        self.computeObservation()

        return self.observation, self.reward, self.done

    def stop(self):
        self.board.close()

    def ack(self):
        self.clientSocket.sendall(b'ack\n')

    def receiveMessage(self):
        msg = b''
        while not msg.endswith(b'\n'):
            msg += self.clientSocket.recv(1024)
        self.currentMessage = msg

    def processBudget(self):
        values = self.currentMessage.split()
        self.timeBudget = int(values[1].decode('utf-8'))
        self.iterationsBudget = int(values[2].decode('utf-8'))

        if DEBUG:
            print(f"NEW BUDGET : {self.timeBudget} {self.iterationsBudget}")
    
        self.ack()

    def processUTT(self):
        if DEBUG:
            print(f"NEW UTT")

        self.unitTypeTable.clear()
        utt = json.loads(self.currentMessage)
        for unitType in utt['unitTypes']:
            self.unitTypeTable.addUnitType(UnitType(unitType))

        self.ack()

    def processPreGameAnalysis(self):
        # Donne une vue d'ensemble de la carte au début du jeu.
        if DEBUG:
            print("PRE GAME ANALYSIS")

        pga = json.loads(self.currentMessage)
        self.terrainWidth = int(pga['pgs']['width'])
        self.terrainHeight = int(pga['pgs']['height'])
        self.terrainMap = pga['pgs']['terrain']

        self.board.resize(self.cell_dimension, (self.terrainHeight, self.terrainWidth))

        self.ack()

    def processGetAction(self):
        # Donne une vue du point de vue du joueur si partiallyObservable est vrai.
        # Donne une vue globale si partiallyObservable est faux.
        if DEBUG:
            print("GET ACTION")

        ga = json.loads(self.currentMessage)
        self.players = []
        for player in ga['pgs']['players']:
            self.players += [Player.fromJSON(player)]
        self.units = []
        self.neutralUnits = []
        self.enemyUnits = []
        for unit in ga['pgs']['units']:
            _unit = Unit(self.unitTypeTable, unit)
            if _unit.player == self.player:
                self.units += [_unit]
            elif _unit.player == -1:
                self.neutralUnits += [_unit]
            else:
                self.enemyUnits += [_unit]

        self.onGoingActions = []
        for action in ga['actions']:
            _action = UnitAction()
            _action.fromJSON(action['action'])
            self.onGoingActions += [[action['ID'], _action]]

        pgs = PhysicalGameState(
            self.players, 
            self.unitTypeTable, 
            self.terrainWidth, 
            self.terrainHeight, 
            self.terrainMap, 
            self.units + self.neutralUnits + self.enemyUnits, 
            self.onGoingActions
        )

        self.availableActions = []
        for unit in self.units:
            alreadyInAction = False
            for onGoingAction in self.onGoingActions:
                if onGoingAction[0] == unit.ID:
                    alreadyInAction = True
                    break
            if not alreadyInAction:
                self.availableActions += [[unit, unit.getUnitActions(pgs)]]

        self.waitForInput = True

    def processMessage(self):
        messages = self.currentMessage.split(b'\n')
        self.currentMessage = messages[-2]

        if messages[0].startswith(b'budget'):
            self.processBudget()
        elif messages[0].startswith(b'utt'):
            self.processUTT()
        elif messages[0].startswith(b'preGameAnalysis'):
            self.processPreGameAnalysis()
        elif messages[0].startswith(b'getAction'):
            self.player = int(str(messages[0], encoding='utf-8').split()[-1])
            self.processGetAction()
        else:
            print(messages)
            raise Exception("Error in env.py : processMessage\nMessage header isn't supported")

    def actionsToAiAction(self, actions: list[list[Unit, UnitAction]]):
        # self.action_space = np.zeros((self.terrainWidth * self.terrainHeight, 7))
        aiAction = np.zeros(self.action_space.shape)

        for unitWithAction in actions:
            unit: Unit          = unitWithAction[0]
            action: UnitAction  = unitWithAction[1]

            if action.type == action.TYPE_MOVE:
                aiAction[unit.x][unit.y][0] = 1
                if action.parameter == action.DIRECTION_RIGHT:
                    aiAction[unit.x][unit.y][1] = 1
                elif action.parameter == action.DIRECTION_DOWN:
                    aiAction[unit.x][unit.y][1] = 2
                elif action.parameter == action.DIRECTION_LEFT:
                    aiAction[unit.x][unit.y][1] = 3
            elif action.type == action.TYPE_HARVEST:
                aiAction[unit.x][unit.y][0] = 2
                if action.parameter == action.DIRECTION_RIGHT:
                    aiAction[unit.x][unit.y][2] = 1
                elif action.parameter == action.DIRECTION_DOWN:
                    aiAction[unit.x][unit.y][2] = 2
                elif action.parameter == action.DIRECTION_LEFT:
                    aiAction[unit.x][unit.y][2] = 3
            elif action.type == action.TYPE_RETURN:
                aiAction[unit.x][unit.y][0] = 3
                if action.parameter == action.DIRECTION_RIGHT:
                    aiAction[unit.x][unit.y][3] = 1
                elif action.parameter == action.DIRECTION_DOWN:
                    aiAction[unit.x][unit.y][3] = 2
                elif action.parameter == action.DIRECTION_LEFT:
                    aiAction[unit.x][unit.y][3] = 3
            elif action.type == action.TYPE_PRODUCE:
                aiAction[unit.x][unit.y][0] = 4
                if action.parameter == action.DIRECTION_RIGHT:
                    aiAction[unit.x][unit.y][4] = 1
                elif action.parameter == action.DIRECTION_DOWN:
                    aiAction[unit.x][unit.y][4] = 2
                elif action.parameter == action.DIRECTION_LEFT:
                    aiAction[unit.x][unit.y][4] = 3
                if action.unitTypeName == "Barracks":
                    aiAction[unit.x][unit.y][5] = 1
                elif action.unitTypeName == "Worker":
                    aiAction[unit.x][unit.y][5] = 2
                elif action.unitTypeName == "Light":
                    aiAction[unit.x][unit.y][5] = 3
                elif action.unitTypeName == "Heavy":
                    aiAction[unit.x][unit.y][5] = 4
                elif action.unitTypeName == "Ranged":
                    aiAction[unit.x][unit.y][5] = 5
            elif action.type == action.TYPE_ATTACK_LOCATION:
                relativeX = action.x - unit.x + self.maxRange
                relativeY = action.y - unit.y + self.maxRange



    def aiActionToActions(self):
        None

    def sample(self):
        actions = []
        for unitActions in self.availableActions:
            for action in unitActions[1]:
                actions += [[unitActions[0], action]]
                
        if len(actions) == 0:
            return None

        return actions[np.random.randint(len(actions))]