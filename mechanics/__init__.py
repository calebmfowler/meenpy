class Force:
    def __init__(self, forceVector, applicationPoint):
        self.force = forceVector
        self.point = applicationPoint
    def __str__(self):
        return f"{self.force} applied at {self.point}"
    
    def getForceVector(self):
        return self.force
    def getApplicationPoint(self):
        return self.point

class FBD:
    def __init__(self, appliedForceSet, reactionForceSet):
        self.forceSet = appliedForceSet
        self.reactionSet = reactionForceSet
    def __str__(self):
        str = "Applied Forces\n"
        for force in self.forceSet:
            str += force.__str__() + "\n"
        str += "Reaction Force Basis\n"
        for reaction in self.reactionSet:
            str += reaction.__str__() + "\n"
        return str[:-1]
    
    def addForces(self, appliedForceSet):
        self.forceSet += appliedForceSet
    def addReactions(self, reactionForceSet):
        self.reactionSet = reactionForceSet
    def getForces(self):
        return self.forceSet
    def getReactions(self):
        return self.reactionSet
    
    def solveReactions(self):
        dofCount = 0
        for reaction in self.reactionSet:
            for component in reaction.getForceVector():
                dofCount += abs(component) >= 1e-6
        if dofCount > 6:
            raise Exception("FBD is not in Static Equilibrium")
        return "Not Finished Developing"