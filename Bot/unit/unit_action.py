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

    def __init__(self):
        self.type = self.TYPE_NONE
        self.x = -1
        self.y = -1
        self.parameter = self.DIRECTION_NONE
        self.unitType = None

    def harvest(self, direction):
        self.type = self.TYPE_HARVEST
        self.parameter = direction

    def produce(self, direction, unitType):
        self.type = self.TYPE_PRODUCE
        self.parameter = direction
        self.unitType = unitType

    def attack(self, x, y):
        self.type = self.TYPE_ATTACK_LOCATION
        self.x = x
        self.y = y

    def toJSON(self):
        json = "{\"type\":" + str(self.type)
        if self.type == self.TYPE_ATTACK_LOCATION:
            json += ", \"x\":" + self.x + ",\"y\":" + self.y
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