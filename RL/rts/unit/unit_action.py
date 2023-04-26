from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rts.unit.unit_type import UnitType

class UnitAction:

    TYPE_NONE = 0
    TYPE_MOVE = 1
    TYPE_HARVEST = 2
    TYPE_RETURN = 3
    TYPE_PRODUCE = 4
    TYPE_ATTACK_LOCATION = 5

    DIRECTION_NONE = -1
    DIRECTION_UP = 0
    DIRECTION_RIGHT = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3

    def __init__(self, type: int = TYPE_NONE, x: int = -1, y: int = -1, parameter: int = DIRECTION_NONE, unitType: UnitType = None):
        self.type = type
        self.x = x
        self.y = y
        self.parameter = parameter
        self.unitType = unitType

    @classmethod
    def typeNone(cls):
        return cls(cls.TYPE_NONE, -1, -1, 1, None)

    @classmethod
    def typeMove(cls, direction):
        type = cls.TYPE_MOVE
        parameter = direction
        return cls(type, -1, -1, parameter, None)


    @classmethod
    def typeHarvest(cls, direction):
        type = cls.TYPE_HARVEST
        parameter = direction
        return cls(type, -1, -1, parameter, None)

    @classmethod
    def typeReturn(cls, direction):
        type = cls.TYPE_RETURN
        parameter = direction
        return cls(type, -1, -1, parameter, None)

    @classmethod
    def typeProduce(cls, direction, unitType):
        type = cls.TYPE_PRODUCE
        parameter = direction
        unitType = unitType
        return cls(type, -1, -1, parameter, unitType)

    @classmethod
    def typeAttack(cls, x, y):
        type = cls.TYPE_ATTACK_LOCATION
        x = x
        y = y
        return cls(type, x, y, cls.DIRECTION_NONE, None)

    def toJSON(self):
        json = "{\"type\":" + str(self.type)
        if self.type == self.TYPE_ATTACK_LOCATION:
            json += ", \"x\":" + str(self.x) + ",\"y\":" + str(self.y)
        else:
            if self.parameter != self.DIRECTION_NONE:
                json += ", \"parameter\":" + str(self.parameter)
            if self.unitType != None:
                 json += ", \"unitType\":\"" + self.unitType.name + "\""
        json += "}"
        return json
    
    def fromJSON(self, json):
        self.type = json['type']
        try:
            self.x = json['x']
            self.y = json['y']
        except:
            None
        try:
            self.parameter = json['parameter']
        except:
            None
        try:
            self.unitType = json['unitType']
        except:
            None

    def toString(self):
        s = "TYPE : "
        if self.type == self.TYPE_NONE:
            s += f"NONE, PARAMETER : {self.parameter}"
        elif self.type == self.TYPE_MOVE:
            s += f"MOVE, PARAMETER : {self.parameter}"
        elif self.type == self.TYPE_HARVEST:
            s += f"HARVEST, PARAMETER : {self.parameter}"
        elif self.type == self.TYPE_RETURN:
            s += f"RETURN, PARAMETER : {self.parameter}"
        elif self.type == self.TYPE_PRODUCE:
            s += f"PRODUCE, PARAMETER : {self.parameter}, UNITTYPE : {self.unitType.name}"
        elif self.type == self.TYPE_ATTACK_LOCATION:
            s += f"ATTACK_LOCATION, X : {self.x}, Y : {self.y}"
        return s