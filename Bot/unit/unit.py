from unit.unit_action import UnitAction

class Unit:

    def __init__(self, unitTypeTable, json):
        self.type = unitTypeTable.find(json['type'])
        self.ID = json['ID']
        self.player = json['player']
        self.x = json['x']
        self.y = json['y']
        self.resources = json['resources']
        self.hitpoints = json['hitpoints']

    def toString(self):
        s = f"ID : {self.ID}, TYPE : {self.type.name}, PLAYER : {self.player},"
        s += f"X : {self.x}, Y : {self.Y}, RESSOURCES : {self.resources}, HITPOINTS : {self.hitpoints}"

    # Retourne la liste d'actions disponible pour l'unité
    # resources, nombre de ressource du joueur
    # terrainDim, un tuple des dimension width et height du terrain
    # terrain, une liste représentant le terrain
    def getUnitActions(self, playerResources, units, terrainDim, terrain):
        actions = []

        uUp = None
        uRight = None
        uDown = None
        uLeft = None

        for unit in units:
            if unit.x == self.x:
                if unit.y == self.y - 1:
                    uUp = unit
                elif unit.y == self.y + 1:
                    uDown = unit
            elif unit.y == self.y:
                if unit.x == self.x - 1:
                    uLeft = unit
                elif unit.x == self.x + 1:
                    uRight = unit

        if unit.type.canAttack:
            if unit.type.attackRange == 1:
                if uUp != None and self.player != uUp.player and uUp.player >= 0:
                    action = UnitAction()
                    action.attack(uUp.x, uUp.y)
                    actions += [action]
                if uRight != None and self.player != uRight.player and uRight.player >= 0:
                    action = UnitAction()
                    action.attack(uRight.x, uRight.y)
                    actions += [action]
                if uDown != None and self.player != uDown.player and uDown.player >= 0:
                    action = UnitAction()
                    action.attack(uDown.x, uDown.y)
                    actions += [action]
                if uLeft != None and self.player != uLeft.player and uLeft.player >= 0:
                    action = UnitAction()
                    action.attack(uLeft.x, uLeft.y)
                    actions += [action]
            else:
                sq_range = unit.type.attackRange * unit.type.attackRange
                for unit in units:
                    if unit.player < 0 or unit.player == self.player:
                        continue
                    sq_dx = (unit.x - self.x) ** 2
                    sq_dy = (unit.y - self.y) ** 2
                    if sq_dx + sq_dy <= sq_range:
                        action = UnitAction()
                        action.attack(unit.x, unit.y)
                        actions += [action]

        if unit.type.canHarvest:
            if self.resources == 0:
                if uUp != None and uUp.type.isResource:
                        action = UnitAction()
                        action.harvest(UnitAction.DIRECTION_UP)
                        actions += [action]
                if uRight != None and uRight.type.isResource:
                        action = UnitAction()
                        action.harvest(UnitAction.DIRECTION_RIGHT)
                        actions += [action]
                if uDown != None and uDown.type.isResource:
                        action = UnitAction()
                        action.harvest(UnitAction.DIRECTION_DOWN)
                        actions += [action]
                if uLeft != None and uLeft.type.isResource:
                        action = UnitAction()
                        action.harvest(UnitAction.DIRECTION_LEFT)
                        actions += [action]
            else:
                None

        for unit in self.type.produces:
            None

        if unit.type.canMove:
            None
    """

            if (resources > 0) {
                if (y > 0 && uup != null && uup.type.isStockpile && uup.player == player) {
                    l.add(new UnitAction(UnitAction.TYPE_RETURN, UnitAction.DIRECTION_UP));
                }
                if (x < pgs.getWidth() - 1 && uright != null && uright.type.isStockpile && uright.player == player) {
                    l.add(new UnitAction(UnitAction.TYPE_RETURN, UnitAction.DIRECTION_RIGHT));
                }
                if (y < pgs.getHeight() - 1 && udown != null && udown.type.isStockpile && udown.player == player) {
                    l.add(new UnitAction(UnitAction.TYPE_RETURN, UnitAction.DIRECTION_DOWN));
                }
                if (x > 0 && uleft != null && uleft.type.isStockpile && uleft.player == player) {
                    l.add(new UnitAction(UnitAction.TYPE_RETURN, UnitAction.DIRECTION_LEFT));
                }
            }
        }

        // if the player has enough resources, adds a produce action for each type this unit produces.
        // a produce action is added for each free tile around the producer 
        for (UnitType ut : type.produces) {
            if (p.getResources() >= ut.cost) {
                int tup = (y > 0 ? pgs.getTerrain(x, y - 1) : PhysicalGameState.TERRAIN_WALL);
                int tright = (x < pgs.getWidth() - 1 ? pgs.getTerrain(x + 1, y) : PhysicalGameState.TERRAIN_WALL);
                int tdown = (y < pgs.getHeight() - 1 ? pgs.getTerrain(x, y + 1) : PhysicalGameState.TERRAIN_WALL);
                int tleft = (x > 0 ? pgs.getTerrain(x - 1, y) : PhysicalGameState.TERRAIN_WALL);

                if (tup == PhysicalGameState.TERRAIN_NONE && pgs.getUnitAt(x, y - 1) == null) {
                    l.add(new UnitAction(UnitAction.TYPE_PRODUCE, UnitAction.DIRECTION_UP, ut));
                }
                if (tright == PhysicalGameState.TERRAIN_NONE && pgs.getUnitAt(x + 1, y) == null) {
                    l.add(new UnitAction(UnitAction.TYPE_PRODUCE, UnitAction.DIRECTION_RIGHT, ut));
                }
                if (tdown == PhysicalGameState.TERRAIN_NONE && pgs.getUnitAt(x, y + 1) == null) {
                    l.add(new UnitAction(UnitAction.TYPE_PRODUCE, UnitAction.DIRECTION_DOWN, ut));
                }
                if (tleft == PhysicalGameState.TERRAIN_NONE && pgs.getUnitAt(x - 1, y) == null) {
                    l.add(new UnitAction(UnitAction.TYPE_PRODUCE, UnitAction.DIRECTION_LEFT, ut));
                }
            }
        }

        // if the unit can move, adds a move action for each free tile around it
        if (type.canMove) {
            int tup = (y > 0 ? pgs.getTerrain(x, y - 1) : PhysicalGameState.TERRAIN_WALL);
            int tright = (x < pgs.getWidth() - 1 ? pgs.getTerrain(x + 1, y) : PhysicalGameState.TERRAIN_WALL);
            int tdown = (y < pgs.getHeight() - 1 ? pgs.getTerrain(x, y + 1) : PhysicalGameState.TERRAIN_WALL);
            int tleft = (x > 0 ? pgs.getTerrain(x - 1, y) : PhysicalGameState.TERRAIN_WALL);

            if (tup == PhysicalGameState.TERRAIN_NONE && uup == null) {
                l.add(new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_UP));
            }
            if (tright == PhysicalGameState.TERRAIN_NONE && uright == null) {
                l.add(new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_RIGHT));
            }
            if (tdown == PhysicalGameState.TERRAIN_NONE && udown == null) {
                l.add(new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_DOWN));
            }
            if (tleft == PhysicalGameState.TERRAIN_NONE && uleft == null) {
                l.add(new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_LEFT));
            }
        }

        // units can always stay idle:
        l.add(new UnitAction(UnitAction.TYPE_NONE, noneDuration));

        return l;
    """