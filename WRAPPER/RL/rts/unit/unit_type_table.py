from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rts.unit.unit_type import UnitType

class UnitTypeTable:

    def __init__(self):
        self.unitTypes: list[UnitType] = []

    def addUnitType(self, unitType: UnitType):
        self.unitTypes += [unitType]

    def find(self, name: str):
        for unitType in self.unitTypes:
            if unitType.name == name:
                return unitType

    def clear(self):
        self.unitTypes = []