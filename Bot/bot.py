import pip
import socket
import json
import numpy as np

from unit.unit import Unit
from unit.unit_action import UnitAction
from unit.unit_type import UnitType
from unit.unit_type_table import UnitTypeTable

from player.player_action import PlayerAction

from board import Board

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
        self.units = []
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
        self.players = ga['pgs']['players']
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

        playerResources = 0
        for player in self.players:
            if player['ID'] == self.player:
                playerResources = player['resources']
                break

        self.units[0].getUnitActions(playerResources, self.units + self.neutralUnits + self.enemyUnits, (self.terrainWidth, self.terrainHeight), self.terrainMap)
        
        # unitAction = UnitAction()
        # unitAction.produce(UnitAction.DIRECTION_UP, self.unitTypeTable.find('Worker'))
        # playerAction.addAction(self.units[0], unitAction)

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

ai = AI()
ai.start()