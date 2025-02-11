class Force:
    def __init__(self, forceVector, applicationPoint):
        self.force = forceVector
        self.point = applicationPoint
    def __str__(self):
        return f"{self.force} applied at {self.point}"