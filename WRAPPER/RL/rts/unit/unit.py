from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rts.unit.unit_type import UnitType
    
from rts.unit.unit_action import UnitAction

from rts.physical_game_state import PhysicalGameState

class Unit:

    def __init__(self, unitTypeTable, json):
        self.type:      UnitType    = unitTypeTable.find(json['type'])
        self.ID:        int         = json['ID']
        self.player:    int         = json['player']
        self.x:         int         = json['x']
        self.y:         int         = json['y']
        self.resources: int         = json['resources']
        self.hitpoints: int         = json['hitpoints']

    def toString(self):
        s = f"ID : {self.ID}, TYPE : {self.type.name}, PLAYER : {self.player},"
        s += f"X : {self.x}, Y : {self.Y}, RESSOURCES : {self.resources}, HITPOINTS : {self.hitpoints}"

    # Retourne la liste d'actions disponible pour l'unité
    # resources, nombre de ressource du joueur
    # terrainDim, un tuple des dimension width et height du terrain
    # terrain, une liste représentant le terrain
    def getUnitActions(self, pgs: PhysicalGameState):
        actions = [UnitAction.typeNone()]

        uUp     = pgs.getUnitAt(self.x, self.y - 1)
        uRight  = pgs.getUnitAt(self.x + 1, self.y)
        uDown   = pgs.getUnitAt(self.x, self.y + 1)
        uLeft   = pgs.getUnitAt(self.x - 1, self.y)

        if self.type.canAttack:
            if self.type.attackRange == 1:
                if uUp != None and self.player != uUp.player and uUp.player >= 0:
                    actions += [UnitAction.typeAttack(uUp.x, uUp.y)]
                if uRight != None and self.player != uRight.player and uRight.player >= 0:
                    actions += [UnitAction.typeAttack(uRight.x, uRight.y)]
                if uDown != None and self.player != uDown.player and uDown.player >= 0:
                    actions += [UnitAction.typeAttack(uDown.x, uDown.y)]
                if uLeft != None and self.player != uLeft.player and uLeft.player >= 0:
                    actions += [UnitAction.typeAttack(uLeft.x, uLeft.y)]
            else:
                sq_range = self.type.attackRange * self.type.attackRange
                for unit in pgs.units:
                    if unit.player < 0 or unit.player == self.player:
                        continue
                    sq_dx = (unit.x - self.x) ** 2
                    sq_dy = (unit.y - self.y) ** 2
                    if sq_dx + sq_dy <= sq_range:
                        actions += [UnitAction.typeAttack(unit.x, unit.y)]

        if self.type.canHarvest:
            if self.resources == 0:
                if uUp != None and uUp.type.isResource:
                        actions += [UnitAction.typeHarvest(UnitAction.DIRECTION_UP)]
                if uRight != None and uRight.type.isResource:
                        actions += [UnitAction.typeHarvest(UnitAction.DIRECTION_RIGHT)]
                if uDown != None and uDown.type.isResource:
                        actions += [UnitAction.typeHarvest(UnitAction.DIRECTION_DOWN)]
                if uLeft != None and uLeft.type.isResource:
                        actions += [UnitAction.typeHarvest(UnitAction.DIRECTION_LEFT)]
            else:
                if uUp != None and uUp.type.isStockpile and uUp.player == self.player:
                        actions += [UnitAction.typeReturn(UnitAction.DIRECTION_UP)]
                if uRight != None and uRight.type.isStockpile and uRight.player == self.player:
                        actions += [UnitAction.typeReturn(UnitAction.DIRECTION_RIGHT)]
                if uDown != None and uDown.type.isStockpile and uDown.player == self.player:
                        actions += [UnitAction.typeReturn(UnitAction.DIRECTION_DOWN)]
                if uLeft != None and uLeft.type.isStockpile and uLeft.player == self.player:
                        actions += [UnitAction.typeReturn(UnitAction.DIRECTION_LEFT)]

        checkUp     = False if uUp    != None else pgs.checkCellAvailability(self.x, self.y - 1)
        checkRight  = False if uRight != None else pgs.checkCellAvailability(self.x + 1, self.y)
        checkDown   = False if uDown  != None else pgs.checkCellAvailability(self.x, self.y + 1)
        checkLeft   = False if uLeft  != None else pgs.checkCellAvailability(self.x - 1, self.y)

        for unitName in self.type.produces:
            unitType = pgs.utt.find(unitName)
            if unitType.cost <= pgs.getPlayerResources(self.player):                
                if checkUp:
                    actions += [UnitAction.typeProduce(UnitAction.DIRECTION_UP, unitType)]
                if checkRight:
                    actions += [UnitAction.typeProduce(UnitAction.DIRECTION_RIGHT, unitType)]
                if checkDown:
                    actions += [UnitAction.typeProduce(UnitAction.DIRECTION_DOWN, unitType)]
                if checkLeft:
                    actions += [UnitAction.typeProduce(UnitAction.DIRECTION_LEFT, unitType)]

        if self.type.canMove:
            if checkUp:
                actions += [UnitAction.typeMove(UnitAction.DIRECTION_UP)]
            if checkRight:
                actions += [UnitAction.typeMove(UnitAction.DIRECTION_RIGHT)]
            if checkDown:
                actions += [UnitAction.typeMove(UnitAction.DIRECTION_DOWN)]
            if checkLeft:
                actions += [UnitAction.typeMove(UnitAction.DIRECTION_LEFT)]
        
        return actions