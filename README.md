## Neural Network

Basic implementation of a feedforward neural network.

### How to use it in a nutshell

![neural-net-rec]
(https://cloud.githubusercontent.com/assets/4638332/18604559/7c637cde-7c7e-11e6-889b-0ab166e675ba.gif)

### Running an execution context for the net

`npm start` to start a node `repl.REPLServer` instance. The `NeuralNet` 
constructor function is already exposed to the REPL, so you can start creating 
your own neural net instances and train them straight away

Much <3 to [Sebastian Gräßl](https://github.com/bastilian) for implementing this

### Create a net instance

`let myNet = new NeuralNet();`

or

```
let customConfig = {
  learningRate: 0.5,
  numEpochs: 2000
};
let myNet = new NeuralNet(customConfig);
```

if you are using a custom configuration.

### Initialise the net

`myNet.initialise([2,3,3,1]);`

Initialising a net instance implies defining the number of layers in the net, and the
number of nodes in each layer.

The code above initialises that net's structure to a 4-layer structure. The input 
layer will consist of 2 neurons, the two hidden layers will have 3 neurons each, and
the output layer just one neuron.

### Train the net

`myNet.train([ [[0,0],[0]], [[0,1],[1]]]);`

Once the net is initialised it can be trained with a given training data set. The
training set can contain one or multiple items, like in the example above. The number
of times the network will train on the given data set is based either on the default 
number of epochs or the one defined in your configuration. This means that for a
number of epochs of 1000 for example, the net will process each entry in the data set
1000 times, before it is finished training. 

### Predict an output

`myNet.predict([[0,1]]);  // hopefully outputs 1 ;)`
