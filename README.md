# neural_network_scratch
Neural Network Library from scratch

## Requirement
>python3, pip3

## Installation

```
git clone https://github.com/diwasx/neural_network_scratch
cd neural_network_scratch
pip3 install -r requirements.txt --user
```
## Library usage
**Creating object and initializing constructor:**

NeuralNetwork (inputLength, hiddenLength, outputLength)
```
n = NeuralNetwork(3, 10, 9)
```

**Generating output using FeedForward**

```
n.feedForward(inputs)
```

**Training NN with know data (supervised learning)**

```
n.trainSVLearing(inputs,targets,learningRate)
```


## Library Implementation (Examples)

* Logical Gates
* Digit Recognition
* Machine Play T-rex game using NN and genetic algorithm (NEAT algorithm) *
