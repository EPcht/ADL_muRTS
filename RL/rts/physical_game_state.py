from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rts.player.player import Player

    from rts.unit.unit import Unit
    from rts.unit.unit_type_table import UnitTypeTable

class PhysicalGameState:

    TERRAIN_NONE = 0
    TERRAIN_WALL = 1

    def __init__(self, players: list[Player], utt: UnitTypeTable, width: int, height: int, terrain: list[int], units: list[Unit]):
        self.players = players
        self.utt = utt
        self.width = width
        self.height = height
        self.terrain = terrain
        self.units = units

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