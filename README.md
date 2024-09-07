# A Neural Network From Scratch

## Description

![Figure_1](https://github.com/Yousefbahr/A-Neural-Network-From-Scratch/assets/101262861/91911a80-6f58-46da-8492-f022ec23cb16)


An implementation of a neural network from scratch using a 
sigmoid activation function and a mean squared error loss function.

- You can either implement your neural network function or use one of the 
general ones that are already implemented.

- A neural network function takes 4 parameters: 
1. **desired_func**: a function that takes two parameters and returns the desired output , which is a single number 
2. **train_data**: the training data that will train the model on, which is a list of x, y coordinates
3. **test_data**: test data to test the model on
4. **performance**: of type 'Performance' has a default of a mean squared error loss function


## Usage


### Input Layer
- Because we are dealing with xy coordinates, the input layer will only consist of two inputs
and the bias. The value of the inputs will change but the bias won't.
- The value of the bias **must be -1**
- 'functions.py' consists of functions to train the neural network on.
- 'data.py' consists of training data
````
import random
from functions import *
from data import *
import matplotlib.pyplot as plt

i1 = InputNeuron("i1", value=1)
i2 = InputNeuron("i2", value=1)
i0 = InputNeuron("Bias", value=-1)
````

---

### Hidden layer

- Initialize neurons in the hidden layer of only type **Neuron**.
- Every neuron has inputs of type **Neuron** or **InputNeuron** and bias input.
- The activation function of every Neuron is the sigmoid function.

```
A = Neuron("A", inputs[i1,i2,i0])
```
---

### Weights

- For every neuron, you add weight using **add_weight** function. 
- **Weight** object takes two parameters of type **InputNeuron** or **Neuron**.This represents the weight connecting the first neuron to the second neuron.
- The weights' values can be pre-initialized or randomly initialized

```
    A.add_weight(Weight(i1, A))
    A.add_weight(Weight(i2, A))
    A.add_weight(Weight(i0, A))
```
---

### Training
- Initialize the net with class **Neural_Net** that takes 4 paramters:
1. **neurons**: List of neurons of type **Neuron** only.
2. **desired**_func: The desired function that takes two inputs and returns one output. There are alot of these functions in 'functions.py'.
3. **performance**: object of type **Performance** which contains the loss function and its derivative.
4. **train_data**: the training data, a list of tuples of xy coordinates

```
net = Neural_Net([A], AND, performance, train_data=logic_operators_data)
net.train()
```
---

### Testing
- **get_output** takes as parameter the testing data and prints accuracy and every test data point with the corresponding model's prediction.

```
    net.get_output(logic_operators_data)
```
---

### Plotting 

- **plot()** takes paramter training data , and plots the positive and negative training data points
and the decision boundary.

```
    net.plot(logic_operators_data)
```
