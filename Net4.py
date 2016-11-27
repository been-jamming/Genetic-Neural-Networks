import math, random
from copy import deepcopy

def MutateNewNeuron(network):
	network.AddNeuron([])
def MutateConnection(network):
	network.Connect(random.randint(0,len(network.net)-1), random.randint(0,len(network.net)-1), random.random())
def MutateChangeWeight(network):
	neuron = random.choice(network.neurons)
	if random.choice((True, False)):
		neuron.weights[random.randint(0, len(neuron.weights)-1)] += random.random()**2
	else:
		neuron.weights[random.randint(0, len(neuron.weights)-1)] -= random.random()**2
def MutateOutputs(network):
	neuron = random.choice(network.neurons)
	neuron.lastout = random.random()
		
mutations = {MutateNewNeuron: 0.01, MutateConnection: 0.3, MutateChangeWeight: 0.6, MutateOutputs: 0.6}

class Neuron():
	def __init__(self, weights = []):
		self.weights = list(weights)
		self.dw = [0]*len(weights)
		self.db = 0
		self.error = 0
		self.lastout = 0
		self.bias = 0
		self.error = 0
		self.inpneurons = []
		self.outneurons = []
	def Eval(self, inputs):
		self.inputs = inputs
		Sum = 0
		for inp, weight in zip(inputs, self.weights):
			Sum += inp*weight
		Sum += self.bias
		self.lastout = 1/(1+math.exp(-Sum))
		return self.lastout
	def Teach(self, u = 0.05, m = 0):
		for w in range(len(self.weights)):
			self.dw[w] = self.inputs[w]*self.error*u + m*self.dw[w]
			self.weights[w] -= self.dw[w]
		self.db = self.error*u + m*self.db
		self.bias -= self.db
	def RandomWeights(self, lower, upper):
		w = len(self.weights)
		self.weights = []
		for i in range(w):
			self.weights.append(random.uniform(lower, upper))
		self.bias = random.uniform(lower, upper)

class Layer():
	def __init__(self, weights):
		self.neurons = []
		for weight in weights:
			self.neurons.append(Neuron(weight))
	def Eval(self, inputs):
		outs = []
		for neuron in self.neurons:
			outs.append(neuron.Eval(list(inputs)))
		return outs
	def Teach(self, u = 0.05, m = 0):
		for neuron in self.neurons:
			neuron.Teach(u, m)
	def CalcErrors(self):
		errors = []
		for r in range(len(self.neurons[0].weights)):
			Sum = 0
			for neuron in self.neurons:
				Sum += neuron.weights[r]*neuron.error
			errors.append(Sum)
		return errors
	def RandomWeights(self, lower, upper):
		for neuron in self.neurons:
			neuron.RandomWeights(lower, upper)

class FeedForward():
	def __init__(self, layers):
		self.layers = []
		for l in range(len(layers)):
			if l == 0:
				self.layers.append(Layer([[1]]*layers[l]))
			else:
				self.layers.append(Layer([[1]*layers[l-1]]*layers[l]))
		self.struct = layers
	def Eval(self, out):
		newo = []
		for neuron, o in zip(self.layers[0].neurons, out):
			newo.append(neuron.Eval([o]))
		out = newo
		L = list(self.layers)
		L.pop(0)
		for layer in L:
			out = layer.Eval(out)
		return out
	def Teach(self, inputs, expected, u=0.05, m = 0, until = False):
		out = self.Eval(inputs)
		errors = [a-e for a,e in zip(out, expected)]
		if until:
			Sum = 0
			for error in errors:
				Sum += error**2
			Sum /= len(errors)
			if Sum < until:
				return False
		for l in reversed(self.layers):
			for n, e in zip(l.neurons, errors):
				n.error = e*n.lastout*(1-n.lastout)
			errors = l.CalcErrors()
		for layer in self.layers:
			layer.Teach(u, m)
		return True
	def TeachFor(self, inputs, num, u=0.05, m = 0, until = False):
		for a in range(num):
			random.shuffle(inputs)
			for inp, expected in inputs:
				if not self.Teach(inp, expected, u, m, until):
					return a
		return True
	def TeachFunc(self, func, num, u = 0.05, m = 0):
		for a in range(num):
			inputs, expected = func()
			self.Teach(inputs, expected, u, m)
	def RandomWeights(self, lower, upper):
		for layer in self.layers:
			layer.RandomWeights(lower, upper)

class Network():
	def __init__(self, net = False):
		if net:
			self.neurons = [Neuron() for neuron in net]
			for neuron, outs in zip(self.neurons, net):
				for out in outs:
					neuron.outneurons.append(self.neurons[out])
					self.neurons[out].inpneurons.append(neuron)
		else:
			self.neurons = []
		self.nextupdates = set([])
		for neuron in self.neurons:
			neuron.weights = [0 for n in neuron.inpneurons]
		self.net = net
	def SetOutput(self, neuron, output):
		self.neurons[neuron].lastout = output
		self.nextupdates = self.nextupdates.union(self.neurons[neuron].outneurons)
	def Step(self, iterations = 1):
		for i in range(iterations):
			next = set([])
			for neuron in self.nextupdates:
				inps = [n.lastout for n in neuron.inpneurons]
				neuron.Eval(inps)
				next = next.union(neuron.outneurons)
			self.nextupdates = next
	def AddNeuron(self, outputs):
		self.neurons.append(Neuron())
		self.net.append(outputs)
		for out in outputs:
			self.neurons[-1].outneurons.append(self.neurons[out])
			self.neurons[out].inpneurons.append(self.neurons[-1])
			self.neurons[out].weights.append(0)
	def Connect(self, out, inp, weight = 0):
		if inp not in self.net[out]:
			self.net[out].append(inp)
		if self.neurons[inp] not in self.neurons[out].outneurons:
			self.neurons[out].outneurons.append(self.neurons[inp])
		if self.neurons[out] not in self.neurons[inp].inpneurons:
			self.neurons[inp].inpneurons.append(self.neurons[out])
			self.neurons[inp].weights.append(weight)
		else:
			self.neurons[inp].weights[self.neurons[inp].inpneurons.index(self.neurons[out])] = weight

def DivNet(network, nummutate = 5):
	new = deepcopy(network)
	for i in range(nummuate):
		mutation = random.choice(mutations)
		if random.random() < mutations[mutation]:
			mutation(new)
	return network, new
			
def Breed(Net1, Net2, genemrate, speciesmrate, numchildren = 2):
	#only works for feed forward neural networks now, not the normal ones
	if Net1.struct != Net2.struct:
		raise Exception("Neural networks are of a different species")
	children = [FeedForward(Net1.struct) for a in range(numchildren)]
	for c in range(len(children)):
		for l in range(len(children[c].layers)):
			for n in range(len(children[c].layers[l].neurons)):
				children[c].layers[l].neurons[n] = random.choice((Net1.layers[l].neurons[n], Net2.layers[l].neurons[n]))
				for w in range(len(children[c].layers[l].neurons[n].weights)):
					children[c].layers[l].neurons[n].weights[w] += random.uniform(-speciesmrate, speciesmrate)
	return children