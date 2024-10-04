# neural_network_scratch
Basic Neural Network Library from scratch.

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

**Neural Network Structure**

```
n.nnStructure()
```

**Training NN with know data (supervised learning)**

```
n.trainSVLearing(inputs,targets,learningRate)
```

**Training Visualization with weights and biases changes**

```
n.trainSVLearingVisualization(inputs,targets,learningRate)
```


## Library Implementation (Examples)

* Logical Gates
* Digit Recognition
* Machine Play T-rex game using NN and genetic algorithm (NEAT algorithm)
* [Disease Prediction](https://github.com/diwasx/diseases_prediction) - Implementation of NN Library to predict the diseases with the given symptoms
