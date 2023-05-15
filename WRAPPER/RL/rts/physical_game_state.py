from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rts.player.player import Player

    from rts.unit.unit import Unit
    from rts.unit.unit_action import UnitAction
    from rts.unit.unit_type_table import UnitTypeTable

class PhysicalGameState:

    TERRAIN_NONE = 0
    TERRAIN_WALL = 1

    def __init__(self, players: list[Player], utt: UnitTypeTable, width: int, height: int, terrain: list[int], units: list[Unit], onGoingActions: list[list[int, UnitAction]]):
        self.players:           list[Player]                = players
        self.utt:               UnitTypeTable               = utt
        self.width:             int                         = width
        self.height:            int                         = height
        self.terrain:           list[int]                   = terrain
        self.units:             list[Unit]                  = units
        self.onGoingActions:    list[list[int, UnitAction]] = onGoingActions

    def getPlayerResources(self, id: int):
        for player in self.players:
            if player.id == id:
                return player.resources
        raise Exception("Error in physical_game_state.py : getPlayerResources\nThe id passed in args doesn't match with any current player id")
    
    def getTerrain(self, x: int, y: int):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return self.TERRAIN_WALL
        else:
            return self.TERRAIN_NONE
        
    def getUnitAt(self, x, y):
        for unit in self.units:
            if unit.x == x and unit.y == y:
                return unit
        return None
        
    def checkCellAvailability(self, x, y):
        # Return True if the cell isn't a wall and if there isn't unit performing an action with it
        # It doesn't check if a unit are already on the cell
        if self.getTerrain(x, y) == self.TERRAIN_WALL:
            return False
        
        for unitWithAction in self.onGoingActions:
            action: UnitAction = unitWithAction[1]
            if action.type == action.TYPE_PRODUCE or action.type == action.TYPE_MOVE:
                for unit in self.units:
                    if unit.ID == unitWithAction[0]:
                        if action.parameter == action.DIRECTION_UP and unit.x == x and unit.y - 1 == y:
                            return False
                        if action.parameter == action.DIRECTION_RIGHT and unit.x + 1 == x and unit.y == y:
                            return False
                        if action.parameter == action.DIRECTION_DOWN and unit.x == x and unit.y + 1 == y:
                            return False
                        if action.parameter == action.DIRECTION_LEFT and unit.x - 1 == x and unit.y == y:
                            return False

        return True
    