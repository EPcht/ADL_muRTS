import pip
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

class AI:
    def __init__(self):
        self.clientSocket = None
        self.timeBudget = -1
        self.iterationsBudget = -1
        self.currentMessage = b''
        self.terrainWidth = 0
        self.terrainHeight = 0
        self.terrainMap = []
        self.player = 0
        self.players = []
        self.unitTypeTable = UnitTypeTable()
        self.units: list[Unit] = []
        self.neutralUnits = []
        self.enemyUnits = []
        self.actions = []
        self.cell_dimension = 30
        self.board = Board()

    def start(self):
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False

        while not connected:
            self.clientSocket.connect((HOST, PORT))
            connected = True

        if DEBUG:
            print("Connected")

        self.ack()

        while True:
            self.receiveMessage()
            self.processMessage()
            self.board.refresh(self.terrainMap, self.units + self.neutralUnits + self.enemyUnits, self.players)
            
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

    # Donne une vue d'ensemble de la carte au début du jeu.
    def processPreGameAnalysis(self):
        if DEBUG:
            print("PRE GAME ANALYSIS")

        pga = json.loads(self.currentMessage)
        self.terrainWidth = int(pga['pgs']['width'])
        self.terrainHeight = int(pga['pgs']['height'])
        self.terrainMap = pga['pgs']['terrain']

        self.board.resize(self.cell_dimension, (self.terrainHeight, self.terrainWidth))

        self.ack()

    # Donne une vue du point de vue du joueur si partiallyObservable est vrai.
    # Donne une vue globale si partiallyObservable est faux.
    def processGetAction(self):
        if DEBUG:
            print("GET ACTION")

        ga = json.loads(self.currentMessage)
        print(ga)
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

        self.actions = []
        for action in ga['actions']:
            _action = UnitAction()
            _action.fromJSON(action['action'])
            self.actions += [[action['ID'], _action]]

        playerAction = PlayerAction()

        pgs = PhysicalGameState(self.players, self.unitTypeTable, self.terrainWidth, self.terrainHeight, self.terrainMap, self.units + self.neutralUnits + self.enemyUnits)

        for unit in self.units:
            alreadyInAction = False
            for action in self.actions:
                if action[0] == unit.ID:
                    alreadyInAction = True
                    break
            if not alreadyInAction:
                availableActions = unit.getUnitActions(pgs)
                if len(availableActions) > 0:
                    playerAction.addAction(unit, np.random.choice(availableActions))

        self.clientSocket.sendall(bytes(playerAction.toJSON(), encoding='utf-8') + b'\n')

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