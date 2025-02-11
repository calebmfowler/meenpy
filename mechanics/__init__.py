class Force:
    def __init__(self, forceVector, applicationPoint):
        self.force = forceVector
        self.point = applicationPoint
    def __str__(self):
        return f"{self.force} applied at {self.point}"

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