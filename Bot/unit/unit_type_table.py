class UnitTypeTable:

    def __init__(self):
        self.unitTypes = []

    def addUnitType(self, unitType):
        self.unitTypes += [unitType]

    def find(self, name):
        for unitType in self.unitTypes:
            if unitType.name == name:
                return unitType

    def clear(self):
        self.unitTypes = []