import socket
import json
import subprocess
import numpy as np
import time

from rts.unit.unit import Unit
from rts.unit.unit_action import UnitAction
from rts.unit.unit_type import UnitType
from rts.unit.unit_type_table import UnitTypeTable

from rts.player.player import Player
from rts.player.player_action import PlayerAction

from rts.physical_game_state import PhysicalGameState

DEBUG = False
HEADLESS = False

if not HEADLESS:
    from env.board import Board

HOST = "127.0.0.1"
PORT = 10000

import threading
class JavaThread (threading.Thread):
    def __init__(self, port: int, mapLocation: str):
        threading.Thread.__init__(self)
        self.port = port
        self.mapLocation = mapLocation

    def run(self):
        subprocess.call(['java', '-jar', '../MicroRTS/bin/MicroRTS.jar', str(self.port), self.mapLocation])

class Env:
    def __init__(self, mapLocation: str):
        # Core variables
        self.mapLocation:       str                                 = mapLocation
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
        self.board:             Board   = None
        if not HEADLESS:
            self.board = Board()

        # For AI
        self.observation:       list[float] = []
        self.observation_space: np.ndarray  = 0
        self.action_space:      np.ndarray  = 0
        self.reward:            float       = 0
        self.done:              bool        = False

        self.winReward:             float = 100
        self.loseReward:            float = -100
        self.harvestReward:         float = 10
        self.returnReward:          float = 10
        self.produceReward:         float = 10
        self.produceBaseReward:     float = 10
        self.produceWorkerReward:   float = 10
        self.produceBarracksReward: float = 10
        self.produceLightReward:    float = 10
        self.produceHeavyReward:    float = 10
        self.produceRangedReward:   float = 10
        self.allyKilledReward:      float = 10
        self.enemyKilledReward:     float = 10

    def start(self):
        JavaThread(PORT, self.mapLocation).start()

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
        
        if not HEADLESS:
            self.board.refresh(self.terrainMap, self.units + self.neutralUnits + self.enemyUnits, self.players)

        self.observation_space = np.zeros((self.terrainWidth, self.terrainHeight, (5 + 5 + 3 + 9 + 6)))
        # In reference to the structuration given in Rapport_ADL_2.pdf page 8.

        self.maxRange = 0
        for unitType in self.unitTypeTable.unitTypes:
            if unitType.attackRange > self.maxRange:
                self.maxRange = unitType.attackRange
        # max_range create a square around each unit to determine the relative position of the attacked unit
        self.action_space = np.zeros((self.terrainWidth, self.terrainHeight, 7))
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
            
    def step(self, aiActions):
        playerAction = PlayerAction()

        if not HEADLESS:
            time.sleep(0.001)

        actions = self.aiActionToActions(aiActions)
        for action in actions:
            playerAction.addAction(action[0], action[1])

        self.clientSocket.sendall(bytes(playerAction.toJSON(), encoding='utf-8') + b'\n')

        self.reward = 0

        self.waitForInput = False
        while not self.waitForInput:
            self.receiveMessage()
            self.processMessage()

        if not HEADLESS:
            self.board.refresh(self.terrainMap, self.units + self.neutralUnits + self.enemyUnits, self.players)

        self.computeObservation()
        return self.observation, self.reward, self.done

    def reset(self):
        if DEBUG:
            print("RESET")

        self.ack()

        time.sleep(0.0001)

        if DEBUG:
            print("SENDED : RESET")
        self.clientSocket.sendall(b'RESET\n')
        
        self.done = False

        self.waitForInput = False

        while not self.waitForInput:
            self.receiveMessage()
            self.processMessage()
        
        self.computeObservation()
        return self.observation

    def stop(self):
        if DEBUG:
            print("STOP")

        self.ack()

        time.sleep(0.0001)

        if DEBUG:
            print("SENDED : STOP")
        self.clientSocket.sendall(b'STOP\n')

        if not HEADLESS:
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
        if DEBUG:
            print("PRE GAME ANALYSIS")

        pga = json.loads(self.currentMessage)
        self.terrainWidth = int(pga['pgs']['width'])
        self.terrainHeight = int(pga['pgs']['height'])
        self.terrainMap = pga['pgs']['terrain']

        if not HEADLESS:
            self.board.resize(self.cell_dimension, (self.terrainHeight, self.terrainWidth))

        self.ack()

    def processGetAction(self):
        if DEBUG:
            print("GET ACTION")

        ga = json.loads(self.currentMessage)
        self.players = []
        for player in ga['pgs']['players']:
            self.players += [Player.fromJSON(player)]

        self.newUnits:      list[Unit]  = []
        self.neutralUnits               = []
        self.newEnemyUnits: list[Unit]  = []

        for unit in ga['pgs']['units']:
            _unit = Unit(self.unitTypeTable, unit)
            if _unit.player == self.player:
                self.newUnits += [_unit]
            elif _unit.player == -1:
                self.neutralUnits += [_unit]
            else:
                self.newEnemyUnits += [_unit]

        # Calcul des rewards
        for newUnit in self.newUnits:
            produced = True
            for lastUnit in self.units:
                if newUnit.ID == lastUnit.ID:
                    produced = False
                    if newUnit.resources > lastUnit.resources:
                        self.reward += self.harvestReward
                    elif newUnit.resources < lastUnit.resources:
                        self.reward += self.returnReward
                    break
            if produced:
                self.reward += self.produceReward
                if newUnit.type.name == "Base":
                    self.reward += self.produceBaseReward
                elif newUnit.type.name == "Worker":
                    self.reward += self.produceWorkerReward
                elif newUnit.type.name == "Barracks":
                    self.reward += self.produceBarracksReward
                elif newUnit.type.name == "Light":
                    self.reward += self.produceLightReward
                elif newUnit.type.name == "Heavy":
                    self.reward += self.produceHeavyReward
                elif newUnit.type.name == "Ranged":
                    self.reward += self.produceRangedReward
        
        for lastUnit in self.units:
            killed = True
            for newUnit in self.newUnits:
                if lastUnit.ID == newUnit.ID:
                    killed = False
                    break
            if killed:
                self.reward += self.allyKilledReward

        for lastEnemyUnit in self.enemyUnits:
            killed = True
            for newEnemyUnit in self.newEnemyUnits:
                if lastEnemyUnit.ID == newEnemyUnit.ID:
                    killed = False
                    break
            if killed:
                self.reward += self.enemyKilledReward

        self.units = self.newUnits
        self.enemyUnits = self.newEnemyUnits

        # ==================

        self.onGoingActions = []
        for action in ga['actions']:
            _action = UnitAction()
            _action.fromJSON(action['action'])
            # Si l'action est de type PRODUCE, retirer le coût de cette production aux ressources actuelles du joueur
            # afin d'éviter des actions interdites
            if _action.type == _action.TYPE_PRODUCE:
                for unit in self.units:
                    if unit.ID == action['ID']:
                        for player in self.players:
                            if player.id == self.player:
                                player.resources -= self.unitTypeTable.find(_action.unitTypeName).cost
                            break
                        break
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
        
        if len(self.availableActions) == 0:
            playerAction = PlayerAction()
            self.clientSocket.sendall(bytes(playerAction.toJSON(), encoding='utf-8') + b'\n')
        else:
            self.waitForInput = True

    def processGameOver(self):
        self.done = True
        self.waitForInput = True
        winner = self.currentMessage.split(b' ')[-1]
        
        if self.player == winner:
            self.reward += self.winReward
        else:
            self.reward += self.loseReward

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
        elif messages[0].startswith(b'gameOver'):
            self.processGameOver()
        else:
            print(messages)
            raise Exception("Error in env.py : processMessage\nMessage header isn't supported")

    def actionsToAiAction(self, actions: list[list[Unit, UnitAction]]):
        aiAction = np.zeros(self.action_space.shape)

        for unitWithAction in actions:
            unit: Unit          = unitWithAction[0]
            action: UnitAction  = unitWithAction[1]

            if action.type == action.TYPE_NONE:
                aiAction[unit.x][unit.y][0] = 0

            elif action.type == action.TYPE_MOVE:
                aiAction[unit.x][unit.y][0] = 1

                if action.parameter == action.DIRECTION_UP:
                    aiAction[unit.x][unit.y][1] = 0
                elif action.parameter == action.DIRECTION_RIGHT:
                    aiAction[unit.x][unit.y][1] = 1
                elif action.parameter == action.DIRECTION_DOWN:
                    aiAction[unit.x][unit.y][1] = 2
                elif action.parameter == action.DIRECTION_LEFT:
                    aiAction[unit.x][unit.y][1] = 3

            elif action.type == action.TYPE_HARVEST:
                aiAction[unit.x][unit.y][0] = 2

                if action.parameter == action.DIRECTION_UP:
                    aiAction[unit.x][unit.y][2] = 0
                elif action.parameter == action.DIRECTION_RIGHT:
                    aiAction[unit.x][unit.y][2] = 1
                elif action.parameter == action.DIRECTION_DOWN:
                    aiAction[unit.x][unit.y][2] = 2
                elif action.parameter == action.DIRECTION_LEFT:
                    aiAction[unit.x][unit.y][2] = 3

            elif action.type == action.TYPE_RETURN:
                aiAction[unit.x][unit.y][0] = 3

                if action.parameter == action.DIRECTION_UP:
                    aiAction[unit.x][unit.y][3] = 0
                elif action.parameter == action.DIRECTION_RIGHT:
                    aiAction[unit.x][unit.y][3] = 1
                elif action.parameter == action.DIRECTION_DOWN:
                    aiAction[unit.x][unit.y][3] = 2
                elif action.parameter == action.DIRECTION_LEFT:
                    aiAction[unit.x][unit.y][3] = 3

            elif action.type == action.TYPE_PRODUCE:
                aiAction[unit.x][unit.y][0] = 4

                if action.parameter == action.DIRECTION_UP:
                    aiAction[unit.x][unit.y][4] = 0
                elif action.parameter == action.DIRECTION_RIGHT:
                    aiAction[unit.x][unit.y][4] = 1
                elif action.parameter == action.DIRECTION_DOWN:
                    aiAction[unit.x][unit.y][4] = 2
                elif action.parameter == action.DIRECTION_LEFT:
                    aiAction[unit.x][unit.y][4] = 3

                if action.unitTypeName == "Base":
                    aiAction[unit.x][unit.y][5] = 0
                elif action.unitTypeName == "Barracks":
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
                aiAction[unit.x][unit.y][0] = 5

                relativeX = action.x - unit.x + self.maxRange
                relativeY = action.y - unit.y + self.maxRange

                aiAction[unit.x][unit.y][6] = relativeX + relativeY * (self.maxRange * 2 + 1)
        
        return aiAction

    def aiActionToActions(self, aiAction: np.ndarray):
        actions: list[Unit, UnitAction] = []
        directions = [UnitAction.DIRECTION_UP, UnitAction.DIRECTION_RIGHT, UnitAction.DIRECTION_DOWN, UnitAction.DIRECTION_LEFT]
        unitTypes = [
            self.unitTypeTable.find("Base"), 
            self.unitTypeTable.find("Barracks"), 
            self.unitTypeTable.find("Worker"), 
            self.unitTypeTable.find("Light"), 
            self.unitTypeTable.find("Heavy"), 
            self.unitTypeTable.find("Ranged"),
        ]

        for x in range(self.terrainWidth):
            for y in range(self.terrainHeight):
                _action = aiAction[x][y]
                # Check if action type is equal to None
                if _action[0] == 0:
                    continue

                find: bool = False
                for unit in self.units:
                    if unit.x == x and unit.y == y:
                        maxType             = np.int64(_action[0])
                        maxMoveParam        = np.int64(_action[1])
                        maxHarvestParam     = np.int64(_action[2])
                        maxReturnParam      = np.int64(_action[3])
                        maxProduceParam     = np.int64(_action[4])
                        maxProduceTypeParam = np.int64(_action[5])
                        maxAttackParam      = np.int64(_action[6])

                        if maxType == 1: # MOVE
                            actions += [[unit, UnitAction.typeMove(directions[maxMoveParam])]]
                        elif maxType == 2: # HARVEST
                            actions += [[unit, UnitAction.typeHarvest(directions[maxHarvestParam])]]
                        elif maxType == 3: # RETURN
                            actions += [[unit, UnitAction.typeReturn(directions[maxReturnParam])]]
                        elif maxType == 4: # PRODUCE
                            actions += [[unit, UnitAction.typeProduce(directions[maxProduceParam], unitTypes[maxProduceTypeParam])]]
                        elif maxType == 5: # ATTACK
                            relativeY = maxAttackParam // (self.maxRange * 2 + 1)
                            relativeX = maxAttackParam - relativeY * (self.maxRange * 2 + 1)
                            _x = unit.x - self.maxRange + relativeX
                            _y = unit.y - self.maxRange + relativeY
                            actions += [[unit, UnitAction.typeAttack(_x, _y)]]
                        find = True
                        break
                if not find:
                    raise Exception("An action was given to a non owned unit.")

        return actions

    def sample(self):
        actions = []

        for unitWithActions in self.availableActions:
            actions += [[unitWithActions[0], unitWithActions[1][np.random.randint(len(unitWithActions[1]))]]]

        return self.actionsToAiAction(actions)
    
    def getMask(self):
        mask = np.zeros((self.terrainWidth, self.terrainHeight, 28 + (self.maxRange * 2 + 1) ** 2), dtype=bool)

        for unitWithActions in self.availableActions:
            _unit: Unit = unitWithActions[0]
            for action in unitWithActions[1]:
                _action: UnitAction = action

                if _action.type == _action.TYPE_NONE:
                    break

                elif _action.type == _action.TYPE_MOVE:
                    mask[_unit.x][_unit.y][1] = True   

                    if _action.parameter == _action.DIRECTION_UP:
                        mask[_unit.x][_unit.y][6] = True
                    elif _action.parameter == _action.DIRECTION_RIGHT:
                        mask[_unit.x][_unit.y][7] = True
                    elif _action.parameter == _action.DIRECTION_DOWN:
                        mask[_unit.x][_unit.y][8] = True
                    elif _action.parameter == _action.DIRECTION_LEFT:
                        mask[_unit.x][_unit.y][9] = True

                elif _action.type == _action.TYPE_HARVEST:
                    mask[_unit.x][_unit.y][2] = True   

                    if _action.parameter == _action.DIRECTION_UP:
                        mask[_unit.x][_unit.y][10] = True
                    elif _action.parameter == _action.DIRECTION_RIGHT:
                        mask[_unit.x][_unit.y][11] = True
                    elif _action.parameter == _action.DIRECTION_DOWN:
                        mask[_unit.x][_unit.y][12] = True
                    elif _action.parameter == _action.DIRECTION_LEFT:
                        mask[_unit.x][_unit.y][13] = True

                elif _action.type == _action.TYPE_RETURN:
                    mask[_unit.x][_unit.y][3] = True   

                    if _action.parameter == _action.DIRECTION_UP:
                        mask[_unit.x][_unit.y][14] = True
                    elif _action.parameter == _action.DIRECTION_RIGHT:
                        mask[_unit.x][_unit.y][15] = True
                    elif _action.parameter == _action.DIRECTION_DOWN:
                        mask[_unit.x][_unit.y][16] = True
                    elif _action.parameter == _action.DIRECTION_LEFT:
                        mask[_unit.x][_unit.y][17] = True

                elif _action.type == _action.TYPE_PRODUCE:
                    mask[_unit.x][_unit.y][4] = True   

                    if _action.parameter == _action.DIRECTION_UP:
                        mask[_unit.x][_unit.y][18] = True
                    elif _action.parameter == _action.DIRECTION_RIGHT:
                        mask[_unit.x][_unit.y][19] = True
                    elif _action.parameter == _action.DIRECTION_DOWN:
                        mask[_unit.x][_unit.y][20] = True
                    elif _action.parameter == _action.DIRECTION_LEFT:
                        mask[_unit.x][_unit.y][21] = True

                    if _action.unitTypeName == "Base":
                        mask[_unit.x][_unit.y][22] = True
                    elif _action.unitTypeName == "Barracks":
                        mask[_unit.x][_unit.y][23] = True
                    elif _action.unitTypeName == "Worker":
                        mask[_unit.x][_unit.y][24] = True
                    elif _action.unitTypeName == "Light":
                        mask[_unit.x][_unit.y][25] = True
                    elif _action.unitTypeName == "Heavy":
                        mask[_unit.x][_unit.y][26] = True
                    elif _action.unitTypeName == "Ranged":
                        mask[_unit.x][_unit.y][27] = True

                elif _action.type == _action.TYPE_ATTACK_LOCATION:
                    mask[_unit.x][_unit.y][5] = True  

                    relativeX = _action.x - _unit.x + self.maxRange
                    relativeY = _action.y - _unit.y + self.maxRange

                    mask[_unit.x][_unit.y][28 + relativeX + relativeY * (self.maxRange * 2 + 1)] = True

        for x in range(self.terrainWidth):
            for y in range(self.terrainHeight):
                mask[x][y][0] = True
                    
        return mask