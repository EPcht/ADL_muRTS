class Player:

    def __init__(self, id: int, resources: int):
        self.id = id
        self.resources = resources

    @classmethod
    def fromJSON(cls, json: str):
        id = json['ID']
        resources = json['resources']
        return cls(id, resources)
    
    def toString(self):
        return f"Player: ID {self.id} / RESOURCES {self.resources}"
