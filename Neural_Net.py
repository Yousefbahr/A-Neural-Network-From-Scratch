import math
import random
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from data import *

def abs_mean(values):
    """Compute the mean of the absolute values a set of numbers.
    For computing the stopping condition for training neural nets"""
    abs_vals = [abs(num) for num in values]
    total = sum(abs_vals)
    return total / float(len(abs_vals))


class Performance:
    """
    For computing the performance of the neural net
    Given two functions , the performance function and its derivative.
    Each function takes two inputs the desired and actual outputs.
    """
    def __init__(self, function = performance_func, derivative = derivative_func):
        self.function = function
        self.derivative = derivative

    def output(self, desired, output):
        return self.function(desired, output)

    def derivative(self, desired, output):
        return self.derivative(desired, output)

class Weight:
    def __init__(self, node1, node2, value=None):
        self.node1 = node1
        self.node2 = node2
        self.value = value

    def __str__(self):
        return f"Weight({self.value}) from {self.node1} to {self.node2} "

    def set_value(self, value):
        self.value = value
    def get_node1_value(self):
        return self.node1.get_value()

    def get_node2_value(self):
        return self.node2.get_value()
    def get_value(self):
        return self.value


class InputNeuron:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return f"Input {self.name}"

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def set_value(self,value):
        self.value = value

class Neuron:
    # Every Neuron has a threshold value
    # 'inputs' is a list of input Nodes
    # 'weights' is a list of Weight instances
    def __init__(self, name, weights=None, inputs=None):
        if weights is None:
            weights = []
        self.name = name
        self.weights = weights
        self.inputs = inputs

    def __str__(self):
        return f"Neuron {self.name}"

    def __repr__(self):
        return f"Neuron {self.name}"

    def add_weight(self,weight):
        # Add weight of type 'Weight' that terminate in the current node
        self.weights.append(weight)

    def get_name(self):
        return self.name

    def get_value(self):
        result = 0
        for weight in self.weights:
            result += weight.get_value() * weight.get_node1_value()

        return 1.0 / (1.0 + math.e ** -result)

class Neural_Net:
    # 'desired_output_func' is a function that takes all the values of the input nodes and returns the desired output
    # performance argument is of type 'Performance' and has both the performance function and its derivative
    def __init__(self, neurons, desired_output_func, performance, train_data):
        self.neurons = neurons
        self.inputs = []
        self.train_data = train_data
        self.weights = []
        self.performance = performance
        self.desired_output_func = desired_output_func
        self.rate = 1
        # add weights and inputs
        for neuron in neurons:
            try:
                for my_input in neuron.inputs:
                    # Must be 'InputNeuron' that is not repeated and not bias input (which is equal to -1)
                    if (isinstance(my_input, InputNeuron) and my_input not in self.inputs) and my_input.value != -1:
                        self.inputs.append(my_input)
            except ValueError:
                exit("Don't add inputs of type 'InputNeuron' in the net")

            # Get output neuron
            if len(self.get_children(neuron)) == 0:
                self.output_neuron = neuron

            # Add weights
            for weight in neuron.weights:
                self.weights.append(weight)

    def get_children(self, node):
        # returns a dictionary of weights and nodes
        # every 'weight' that connects this 'node' to all other nodes
        # returns empty dict if 'node' is an output node
        result = {}
        for weight in self.weights:
            if weight.node1 == node:
                result.update({weight: weight.node2})
        return result

    def delta(self, node):
        # final layer
        if self.output_neuron == node:
            input_nodes_values = []
            # Get input nodes values
            for my_node in self.inputs:
                input_nodes_values.append(my_node.get_value())

            # I treated the 'actual output' as the output of the final node
            return node.get_value() * (1 - node.get_value()) * self.performance.derivative(self.desired_output_func(input_nodes_values) , node.get_value())

        # Not final layer
        else:
            result = node.get_value() * (1 - node.get_value())
            other_sum = 0
            for weight, node_child in self.get_children(node).items():
                other_sum +=  weight.get_value() * self.delta(node_child)
            return result * other_sum

    def train(self, target_performance = 0.0001, random_weights = True):
        random.seed(42)
        # Set random numbers for weights (0 or 1)
        if random_weights:
            for weight in self.weights:
                weight.set_value(random.randrange(-1, 3))
                print(f"{weight} initally, {weight.get_value()}")

        iteration = 0
        while iteration < 10000:
            performances = []
            for j, datum in enumerate(self.train_data):

                for i in range(len(self.inputs)):
                    self.inputs[i].set_value(datum[i])

                for weight in self.weights:
                    new_delta = self.delta(weight.node2)
                    new_weight = self.rate * weight.get_node1_value() * new_delta
                    weight.set_value(weight.get_value() + new_weight)

                desired = self.desired_output_func(datum)
                output = self.output_neuron.get_value()
                performance = self.performance.output(desired, output)
                performances.append(performance)

            if iteration % 1000 == 0:
                abs_mean_performance = abs_mean(performances)
                print(f"after {iteration} iterations performance is: {abs_mean_performance}")
                if abs_mean_performance <= target_performance:
                    print(f"Target performance exceeded {abs_mean_performance}")
                    break
            iteration += 1

        print()
        for weight in self.weights:
            print(f"{weight} finally, {weight.get_value()}")

    def get_output(self, test_data):
        correct_ans = 0
        total = len(test_data)
        for my_inputs in test_data:

            # change inputs with specified new values
            for i, node in enumerate(self.inputs):
                node.value = my_inputs[i]

            # get output of final node
            result = self.output_neuron.get_value()

            correct = self.desired_output_func(my_inputs)
            if correct == round(result):
                correct_ans += 1

            print(f"test {my_inputs} -> {result}, correct answer -> {correct}")

        print(f"Accuracy = {correct_ans / total}")

    def plot(self, data, x_coord = 0, y_coord = 5):
        """
        'data' : the positive and negative samples
        x_coord , y_coord: are the x and y coordinates to start from to draw the lines of the weights
        Plot the lines (using the weights) of the neurons of the first hidden layer
        and plot the positive and negative 'train_data' data points
        Made only for two input nodes
        """
        first_layer_neurons = []
        missing_xaxis = False
        for neuron in self.neurons:
            # Must be only the first layer of Neurons
            # which have the nets' inputs as the neuron's inputs
            # dealing with the inputs of the net as only two inputs
            if self.inputs[0] in neuron.inputs or self.inputs[1] in neuron.inputs:
                first_layer_neurons.append(neuron)
                weights = []
                # get weights connected to the current neuron
                for weight in neuron.weights:
                    # get bias weight
                    if weight.get_node1_value() == -1:
                        bias = weight.value
                    # not bias
                    else:
                        if (weight.node1.name == 'i2'):
                            missing_xaxis = True

                        weights.append(weight.value)

                # neuron connected with two weights
                if len(weights) > 1:
                    x = np.arange(x_coord, y_coord)
                    y = (-weights[0] * x + bias) / weights[1]

                # less than two weights connected
                else:
                    # Horizontal line
                    if missing_xaxis:
                        x = np.arange(x_coord, y_coord)
                        y = np.full((len(x),), bias / weights[0])
                    # Vertical line
                    else:
                        y = np.arange(x_coord, y_coord)
                        x = np.full((len(y),), bias / weights[0])

                plt.plot(x, y)

        # Plot positive and negative data samples
        positive = list(zip(*[datum for datum in data if self.desired_output_func(datum) == 1]))
        negative = list(zip(*[datum for datum in data if self.desired_output_func(datum) == 0]))
        plt.scatter(positive[0], positive[1])
        plt.scatter(negative[0], negative[1])

        plt.legend(first_layer_neurons + ["Positive", "Negative"] )
        plt.show()


def basic_net(desired_func, train_data, test_data, performance=Performance()):
    # Basic net made from one neuron and two inputs

    i1 = InputNeuron("i1", value=1)
    i2 = InputNeuron("i2", value=1)
    # The Bias is the  same with all neurons, and its value doesn't change
    # Must be named Bias
    i0 = InputNeuron("Bias", value=-1)

    A = Neuron("A", inputs=[i1, i2, i0])

    A.add_weight(Weight(i1, A))
    A.add_weight(Weight(i2, A))
    A.add_weight(Weight(i0, A))

    # The Bias is not inserted in the net
    # Only of type 'Neuron' is inserted in the net
    net = Neural_Net([A], desired_func, performance,
                     train_data=train_data)
    net.train()

    net.get_output(test_data)

    net.plot(train_data)


def two_layer_net(desired_func,train_data, test_data, performance=Performance()):
    # A Two layer net made from three neurons and two inputs

    i1 = InputNeuron("i1", value=1)
    i2 = InputNeuron("i2", value=1)
    # The Bias is the same with all neurons, and its value doesn't change
    i0 = InputNeuron("Bias", value=-1)

    A = Neuron("A", inputs=[i1, i2, i0])
    B = Neuron("B", inputs=[i1, i2, i0])
    C = Neuron("C", inputs=[A, B, i0])

    A.add_weight(Weight(i1, A))
    A.add_weight(Weight(i2, A))
    A.add_weight(Weight(i0, A))

    B.add_weight(Weight(i1, B))
    B.add_weight(Weight(i2, B))
    B.add_weight(Weight(i0, B))

    C.add_weight(Weight(A, C))
    C.add_weight(Weight(B, C))
    C.add_weight(Weight(i0, C))

    # The Bias is not inserted in the net
    # only 'Neurons' are inserted, not even the inputs
    net = Neural_Net([A, B, C],  desired_func, performance ,
                     train_data=train_data)
    net.train()

    net.get_output(test_data)

    net.plot(train_data)

def multi_neuron_net(desired_func, train_data, test_data, performance=Performance()):
    # A Two layer net made from four neurons and two inputs
    i1 = InputNeuron("i1", value=1)
    i2 = InputNeuron("i2", value=1)
    # The same with all neurons, and its value doesn't change
    i0 = InputNeuron("Bias", value=-1)

    A = Neuron("A", inputs=[i1, i2, i0])
    B = Neuron("B", inputs=[i1, i2, i0])
    C = Neuron("C", inputs=[i1, i2, i0])

    D = Neuron("D", inputs=[A, B, C ,i0])

    A.add_weight(Weight(i1, A))
    A.add_weight(Weight(i2, A))
    A.add_weight(Weight(i0, A))

    B.add_weight(Weight(i1, B))
    B.add_weight(Weight(i2, B))
    B.add_weight(Weight(i0, B))

    C.add_weight(Weight(i1, C))
    C.add_weight(Weight(i2, C))
    C.add_weight(Weight(i0, C))

    D.add_weight(Weight(A, D))
    D.add_weight(Weight(B, D))
    D.add_weight(Weight(C, D))
    D.add_weight(Weight(i0, D))

    # The Bias is not inserted in the net
    # only of type 'Neurons' are inserted, not even the inputs
    net = Neural_Net([A, B, C, D],  desired_func, performance,
                     train_data=train_data)
    net.train()

    net.get_output(test_data)

    net.plot(train_data)


if __name__ == "__main__":
    # basic_net(AND, data, data)
    # two_layer_net(EQUAL,train_data=data, test_data=data)

    # two_layer_net(square, train_data=square_data, test_data=square_data)

    # multi_neuron_net(square,train_data=square_data, test_data=square_data)
    multi_neuron_net(donut, train_data=donut_data, test_data=donut_data)

    # multi_neuron_net(EQUAL,train_data=data, test_data=data)

    # basic_net(NAND)
    # basic_net(OR)

