from numpy import exp, dot, ones

class InputNeuron:
    def __init__(self):
        self.input = 0
    def setOutput(self, input):
        self.output = input
    def getOutput(self):
        return self.output

class Neuron:
    def __init__(self, weights, inputNeurons):
        self.weights = weights
        self.inputNeurons = inputNeurons
        self.output = 0
    def getOutput(self):
        return self.output
    def stimulate(self):
        sigmoid = lambda s : 1 / (1 + exp(-s))
        self.output = int(sigmoid(dot(self.weights, [neuron.output for neuron in self.inputNeurons])) >= 0.5)

class Net:
    def __init__(self, shape):
        self.neurons = []
        for i in range(len(shape)):
            self.neurons += []
            for j in range(shape[i]):
                if i == 0:
                    self.neurons[i] += InputNeuron()
                else:
                    self.neurons[i] += Neuron(
                        ones((len(self.neurons[i - 1]),)),
                        self.neurons[i - 1]
                    )
    def setInputs(self, inputs):
        for i in range(len(self.neurons[0])):
            self.neurons[0, i].setOutput(inputs[i])
    def eval(self):
        for i in range(1, len(self.neurons)):
            for j in range(len(self.neurons[i])):
                self.neurons[i, j].stimulate()
        return [neuron.getOutput() for neuron in self.neurons[-1]]