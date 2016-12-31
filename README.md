## Neural Network

Basic implementation of a feedforward neural network.

### How to use it in a nutshell

**WARNING! Needs updating. Please refer to the rest of the readme for updated APIs**

![neural-net-rec]
(https://cloud.githubusercontent.com/assets/4638332/18604559/7c637cde-7c7e-11e6-889b-0ab166e675ba.gif)

### Running an execution context for the net

Run `npm start` to start a node `repl.REPLServer` instance. The `NeuralNet` 
constructor function is already exposed to the REPL, so you can start creating 
your own neural net instances and train them straight away

Much <3 to [Sebastian Gräßl](https://github.com/bastilian) for implementing this

### Cleaning the snapshots folder

Run `npm run clean-snapshots` to remove all snapshot files in the snapshots folder

### Create a net instance

`let myNet = new NeuralNet();`

or

```javascript
let customConfig = {
  learningRate: 0.5,
  numEpochs: 2000
};
let myNet = new NeuralNet(customConfig);
```

if you are using a custom configuration.

### Initialise the net based on a layer configuration

```javascript
myNet.initialise({layers: [2,3,3,1]});
```

Initialising a net instance using a layer configuration implies defining the number 
of layers in the net, and the number of nodes in each layer.

The code above initialises that net's structure to a 4-layer structure. The input 
layer will consist of 2 neurons, the two hidden layers will have 3 neurons each, and
the output layer just one neuron.

### Initialise the net based on an existing snapshot

An alternative to the above method is to initialise the net with data from an existing
snapshot. 

```javascript
let myNet = new NeuralNet();
myNet.initalise('path/to/snapshot.json');
myNet.predict([0,1]);
```

A snapshot is a .json file which contains the configuration details of a pre-trained 
net. The file contains a weights & biases configuration for each node on each 
layer, which will be used to initialise the nodes of the net. For a sample of the expected 
json object format, please check the 
[sample snapshot](https://github.com/CarmenPopoviciu/neural-net/blob/master/sample-snapshot.json).

Snapshots will prove to be extremely useful for use cases when one wants to train a
net from an already pre-trained state. Snapshots also open up the possibilities
as to where the actual net training needs to happen. One such example would be having
a net instance trained in the cloud and then porting its resulting weights and biases
to a local net.
 

### Train the net

```javascript
myNet.train([ [[0,0],[0]], [[0,1],[1]]]);
```

Once the net is initialised it can be trained with a given training data set. The
training set can contain one or multiple items, like in the example above. The number
of times the network will train on the given data set is based either on the default 
number of epochs or the one defined in your configuration. This means that for a
number of epochs of 1000 for example, the net will process each entry in the data set
1000 times, before it is finished training. 

### Predict an output

```javascript
myNet.predict([0,1]);  // hopefully outputs 1 ;)
```
