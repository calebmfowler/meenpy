class FBD:
    def __init__(self, appliedForceSet, reactionForceSet):
        self.forceSet = appliedForceSet
        self.reactionSet = reactionForceSet
    def __str__(self):
        str = "Applied Forces\n"
        str << [force.__str__() for force in self.forceSet]
        str = "Reaction Force Basis\n"
        str << [reaction.__str__() for reaction in self.reactionSet]
        return str