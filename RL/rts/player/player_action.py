from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rts.unit.unit import Unit
    from rts.unit.unit_action import UnitAction

class PlayerAction:

    def __init__(self):
        self.actions = []

    def addAction(self, unit: Unit, unitAction: UnitAction):
        self.actions.append([unit, unitAction])

    def toJSON(self):
        first = True
        json = "["

        for pair in self.actions:
            unit = pair[0]
            unitAction = pair[1]
            if not first:
                json += " ,"
            json += "{\"unitID\":" + str(unit.ID) + ", \"unitAction\":"
            json += unitAction.toJSON()
            json += "}"
            first = False

        json += "]"
        return json