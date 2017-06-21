(function() {
  'use strict';
  const Neuron = require('./neuron');
  const activations = require('./activations');
  const mse = require('./utils/mse');
  const jsonFile = require('jsonfile');

  /**
   * Very basic and minimal implementation of an artiffical neural network. The net
   * only supports the sigmoid function as the activation function and has no
   * implemented optimization algorithms at the time being (that will come later ;))
   * 
   * 
   * For more details about what a neural network is, see 
   * (https://en.wikipedia.org/wiki/Artificial_neural_network) and google, google,
   * google ;)
   * 
   * Upon instantiation, the net can be configured with the following 
   * properties: 
   *   1. a LEARNING RATE, which indicates how fast a neuron will learn
   *   2. the NUMBER OF EPOCHS, which is a measure of the number of times the
   *      entire training data set is fed to the network in order to train it
   * 
   * The network exposes five main APIs:
   *   1. `initialise`, which initialses the structure of the net with a given number of
   *       layers and nodes per layer (see config param for more details on the 
   *       expected data type), or based on a pre-existing snapshot
   *   2. `addLayer`, which adds a layer with a given number of nodes to the net
   *   3. `train`, which trains the net given some training data and based on its
   *       configuration values
   *   4. `predict`, which predicts the net's output, given a specific input
   *   5. `gradientCheck`, which performs numerical gradient checking. Do note that 
   *      numerical gradient checking is a way to test the correctness of the computations
   *      of the network and therefore requires a different workflow. In order to perform
   *      numerical gradient checking, one first needs to train the net with one training
   *      data item and only after that perform gradient checking.
   *   6. `takeSnapshot`, which takes a snapshot of the net's current configuration,
   *      including number of layers, nodes ina layer, weights and biases of each node
   *      in each layer. See `snapshot-sample.json` for an example of output file
   * 
   * This NN implementation allows new instances of neural nets to be created, which
   * can then be initialised with a predefined layer structure (to which new 
   * layers can be afterwards added):
   * 
   *   // create net with 3 layers and 1 neuron on first, 2 on the second and 3 neurons
   *   // on the third layer
   *   let myNet = new NeuralNet();
   *   myNet.initialise({layers: [1,2,3]}) 
   *   myNet.addLayer(1); // add a fourth layer with 1 neuron
   * 
   * or with a layer configuration predefined in a snapshot file
   * 
   *  // create net with a layers configuration as defined in snapshot.js
   *  let myNet = new NeuralNet();
   *  myNet.initialise({snapshot: 'path/to/snapshot.json'});
   * 
   * or with no layer configuration, to which new layers can be imperatively added:
   * 
   *   let myNet2 = new NeuralNet();
   *   myNet2.addLayer(2); // add input layer with 2 neurons
   *   myNet2.addLayer(4); // add hiddent layer with 4 neurons 
   *   ...
   * 
   * Training a net requires a training data set, which is passed as an argument to the
   * `train` API. The current implementation does not support pre-processing of predefined 
   * data sets, such as the MNIST data set (or others). In order to use such training sets
   * one must process these entries separately and pass the result to the net's `train` API.
   * 
   * For example, a training set used in teacing the network how to compute logical XOR, looks
   * like this:
   *   let xorTrainingSet = [[[0,0],[0]], [[0,1],[1]], [[1,0],[1]], [[1,1],[]0]];
   *   net.train(xorTrainingSet); 
   * 
   * 
   * @param {Object} config              The net's configuration
   * @param {Array}  config.layers       The layers in the net. The length of the Array
   *                                     will represent the number of layers and the array 
   *                                     values will represent the number of corresponding
   *                                     nodes in each layer  
   * @param {Number} config.learningRate The learning rate
   * @param {Number} config.numEpochs    The number of epochs used for training the net 
   */
  function NeuralNet(config) {
    config = config || {};

    this.layers = [];
    this.batchSize = config.batchSize || 1;
    this.learningRate = config.learningRate || 0.3;
    this.numEpochs = config.numEpochs || 10000;
  }

  /**
   * @description
   * Initialise the network with a given layer structure or based on a pre-existing snapshot. 
   * 
   * A layer structure is a simple array, whose length represents the number of layers in the 
   * net, and whose values represent the number of neurons per each layer
   * 
   * For example
   *   let net = new NeuralNet();
   *   net.initialise({layers: [2,3,1]});
   * 
   * creates a neural net with 3 layers and 2 neurons on the first layer, 3 on the
   * second and one neuron on the output layer. All weights and biases in the net will
   * be randomly generated with values between 0 and 1 in the initialisation process
   * 
   * A snapshot is a .json file which contains the configuration details of a 
   * pre-trained net. The file contains the weights and biases for each node on each 
   * layer of the network, which will be used to initialise the nodes of the net. 
   * For a sample of the expected json object format, please check 'snapshot-sample.json' 
   * in the root of this project
   * 
   * For example
   *  let net = new NeuralNet();
   *  net.initialise({snapshot: 'path/to/snapshot.json'});
   * 
   * creates a neural net with a layer structure, weights and biases as defined in the 
   * provided json file
   * 
   * If both 'config.layers' and 'config.snapshot' are provided, 'config.snapshot' has precedence
   * and the net will be initialised based on the provided snapshot
   * 
   * 
   * @param {Object} config The network configuration
   * @param {Array}  config.layers               The layers configuration
   * @param {String | Object} config.snapshot    Snapshot filename or object
   * @param {String} config.activation           Name of the activation function. See activations.js for 
   *                                             supported functions   
   */
  NeuralNet.prototype.initialise = function(config) {
    if (!config.layers && !config.snapshot) {
      throw new Error(
        `initialise: Can't initialise net. Please provide a layer configuration or a snapshot to get started`
      );
    }

    if (
      config.activation &&
      !activations.isSupportedActivation(config.activation)
    ) {
      throw new Error(
        `initialise: Unsupported activation function. Please use one of the following: ${activations.activations()}`
      );
    }

    if (config.snapshot) initialiseNetFromSnapshot.apply(this, [config]);
    else initialiseNetWithLayerConfig.apply(this, [config]);
  };

  /**
   * @description
   * Add a new layer, with a given number of nodes, to the network
   * 
   * @param {Number} numNodes    Number of nodes in the layer
   * @param {String} activation  The activation function for each node in the layer
   * @param {Array} [nodeValues] Initialisation values for each node in the layer
   */
  NeuralNet.prototype.addLayer = function addLayer(
    numNodes,
    activation,
    nodeValues
  ) {
    if (!numNodes)
      throw new Error(
        `addLayerFn: Please specify the number of nodes in the layer.`
      );
    if (typeof numNodes !== 'number' || !Number.isInteger(numNodes)) {
      throw new Error(
        `addLayerFn: Number of nodes arg should be an integer, got ${typeof numNodes} instead.`
      );
    }

    this.layers.push({
      nodes: initialiseNodes(numNodes, this.layers, activation, nodeValues)
    });
  };

  /**
   * @description
   * The net training algorithm. This is based on feedforward and backpropagation 
   * with stochastic gradient descent. 
   * 
   * The algorithm has support for both training in batches, or per data set entry instead. This means 
   * that, the net will readjust its weight and biases either after each batch or each entry in the training 
   * data set is processed by it.
   * 
   * For more details about 
   *   - feedforward, see (https://en.wikipedia.org/wiki/Feedforward_neural_network)
   *   - backpropagation, see (http://neuralnetworksanddeeplearning.com/chap2.html)
   *   - gradient descent, see (http://sebastianruder.com/optimizing-gradient-descent/)
   * 
   * @param {Array} data The training data
   * 
   * TODO(@carmen) the format of the expected training data set sucks. Fix it!!
   */
  NeuralNet.prototype.train = function(trainData) {
    let crrEpoch = 1;

    while (crrEpoch <= this.numEpochs) {
      trainData.forEach((set, index) => {
        let input = set[0];
        let output = set[1];

        feedforward(input, this.layers);
        backprop(output, this.layers);
        if ((index + 1) % this.batchSize === 0) {
          updateWeights(this.layers, this.learningRate);
          updateBiases(this.layers, this.learningRate);
        }
      });
      crrEpoch++;
    }
  };

  /**
   * @description
   * The prediction function. Predicts the output for a given input
   * 
   * @param {Array} data The input set
   * @return {Array} The predicted output 
   */
  NeuralNet.prototype.predict = function(data) {
    return feedforward(data, this.layers);
  };

  /**
   * @description
   * Take a snaphot of the current net configuration(layers, nodes per layer and
   * the weights and bias for each node) and save them in a json file
   */
  NeuralNet.prototype.takeSnapshot = function() {
    let now = new Date();
    let date = `${now
      .toLocaleDateString()
      .replace(/\//g, '-')}-${now.toLocaleTimeString()}`;
    let file = `${__dirname}/snapshots/snapshot(${date}).json`;
    let snapshot = { layers: [] };

    this.layers.forEach(layer => {
      let crrLayerSnapshot = {
        nodes: [],
        activation: layer.activation
      };

      layer.nodes.forEach(function(node) {
        crrLayerSnapshot.nodes.push({
          weights: node.weights,
          bias: node.bias
        });
      });

      snapshot.layers.push(crrLayerSnapshot);
    });

    jsonFile.writeFile(file, snapshot, { spaces: 2, flag: 'wx' }, function(
      err
    ) {
      if (err) console.log(`takeSnapshot: ${err}`);
    });
  };

  /**
   * @description
   * Perform numerical gradient check to verify if the gradients computed by the net
   * are correct. 
   * 
   * For more details on numerical gradient check, 
   * see (http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)
   * 
   * @param {Array} trainData The train data set
   *
   */
  NeuralNet.prototype.gradientCheck = function(trainData) {
    let lyrs = this.layers.slice(1, this.layers.length), // exclude first layer since it has no weights
      input = trainData[0],
      expectedOutput = trainData[1],
      epsilon = Math.pow(10, -4),
      numGradient;

    lyrs.forEach((layer, layerIndex) => {
      layer.nodes.forEach((node, nodeIndex) => {
        let numericalGradients = [];
        node.weights.forEach((weight, index) => {
          node.weights[index] = weight + epsilon;
          let out1 = this.predict(input);
          let loss1 = mse(expectedOutput, out1);

          node.weights[index] = weight - epsilon;
          let out2 = this.predict(input);
          let loss2 = mse(expectedOutput, out2);

          numGradient = (loss1 - loss2) / (2 * epsilon);
          numericalGradients.push(numGradient);

          // reset weight to original value
          node.weights[index] = weight;
        });

        console.log(
          `The numerical gradients for node ${nodeIndex +
            1} on layer ${layerIndex + 2} are: ${numericalGradients}`
        );
        console.log(
          `The computed gradients for the same nodes are: ${node.gradientsW}`
        );
        console.log();
      });
    });
  };

  /**
   * @description
   * Initialise net based on a given layers configuration (see the `initialise` js doc
   * for a complete description of what a snapshot is)
   * 
   * @param {Object} config The net configuration object (see `initialise` function
   *                        js docs for a complete description of the expected object)
   */
  function initialiseNetWithLayerConfig(config) {
    if (!config.layers.length) {
      throw new Error(
        `initialiseNetWithLayerConfig: Can't initialise net with undefined number of layers`
      );
    }
    config.layers.forEach(numNodes =>
      this.addLayer(numNodes, config.activation)
    );
  }

  /**
   * @description
   * Initialise net based on 
   *   - an external snapshot file (see the `initialise` js doc for a complete 
   *     description of what a snapshot is)
   *   - a snapshot-like object
   * 
   * @param {Object} config The NN configuration
   */
  function initialiseNetFromSnapshot(config) {
    let addLayersFromSnapshot = snapshot => {
      snapshot.layers.forEach(layer => {
        this.addLayer(layer.nodes.length, layer.activation, layer.nodes);
      });
    };

    if (typeof config.snapshot === 'string') {
      jsonFile.readFile(config.snapshot, (err, obj) => {
        if (err) console.log(err);
        addLayersFromSnapshot(obj);
      });
    } else {
      if (config.snapshot && typeof config.snapshot === 'object') {
        addLayersFromSnapshot(config.snapshot);
      }
    }
  }

  /**
   * @description
   * Initialise a given number of neurons(referred here as 'nodes' for brevity) 
   * that will be part of a given layer in the net. In this context, the process of
   * initialisation refers to creating neuron instances and initialising their 
   * weights, biases, and error. 
   * 
   * @param {Number} numNodes The number of neurons to initialise
   * @param {Array} layers The net's layers
   * @param {Array} [values] Values to initialise nodes with
   * @return {Array} The initialised neurons
   */
  function initialiseNodes(numNodes, layers, activation, values = []) {
    let nodes = [];

    for (let i = 0; i < numNodes; i++) {
      let node = new Neuron();
      if (layers.length > 0) {
        values[i] = values[i] || {};
        // we only need to initialise weigths/biases/error for the hidden/output layers
        node.initialiseWeights(
          layers[layers.length - 1].nodes.length,
          values[i].weights
        );
        node.initialiseBias(values[i].bias);
        node.initialiseError();
        node.initialiseActivationFn(activation);
      }
      nodes.push(node);
    }
    return nodes;
  }

  /**
   * @description
   * The feedforward algorithm. For more details, see 
   * (https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Architecture/feedforward.html)
   * 
   * @param {Array} inputs The training inputs
   * @return {Array} The network's predicted output 
   */
  function feedforward(inputs, layers) {
    let output = [];
    layers.forEach(function(crrLayer, layerIndex) {
      crrLayer.nodes.forEach(function(node, index) {
        if (layerIndex === 0) {
          // input layer
          node.output = inputs[index];
        } else {
          // hidden layers
          node.updateWeightedInput(layers[layerIndex - 1]);
          node.updateOutput();

          if (layerIndex === layers.length - 1) {
            //output layer
            output.push(node.output);
          }
        }
      });
    });

    return output;
  }

  /**
   * @description
   * The backpropagation algorithm. For details, see 
   * (https://en.wikipedia.org/wiki/Backpropagation)
   * 
   * This implementation is based on the 4 BP equations as defined in
   * (http://neuralnetworksanddeeplearning.com/chap2.html)
   * 
   * @param {Array} expectedOutput The expected output for a given training set
   */
  function backprop(expectedOutput, layers) {
    // compute error(delta_L) for output layer
    // delta[L] = (predictedOutput-expectedOutput)*sigmoidPrime(weightedInput[L])
    let outputLayer = layers[layers.length - 1];
    computeOutputLayerError(outputLayer, expectedOutput);

    // compute error(delta_l) for hidden layers
    // delta[l] = (w[l+1]*delta[l+1])*sigmoidPrime(weightedInput[l])
    computeHiddenLayersErrors(layers);
  }

  /**
   * @description
   * Implement first BP equation as described in 
   * (http://neuralnetworksanddeeplearning.com/chap2.html)
   * 
   * This equation provides a way to compute the error in the output layer
   * 
   * @param {Object} outputLayer The output layer
   * @param {Array} expectedOutput The net expected output for a given training input
   */
  function computeOutputLayerError(outputLayer, expectedOutput) {
    outputLayer.nodes.forEach(function(node, nodeIndex) {
      node.updateOutputError(expectedOutput[nodeIndex]);
    });
  }

  /**
   * @description
   * Implement the second BP equation, as described in
   * (http://neuralnetworksanddeeplearning.com/chap2.html)
   * 
   * This equation provides a way to compute the error in each neuron on layer l,
   * with respect to the error in each neuron of the next layer (l+1)
   */
  function computeHiddenLayersErrors(layers) {
    let crrLayer, nextLayer;

    // we want to compute this for all layers except input/output layers
    for (let i = layers.length - 2; i > 0; i--) {
      crrLayer = layers[i];
      crrLayer.nodes.forEach(function(node, crrNodeIndex) {
        nextLayer = layers[i + 1];
        node.updateError(crrNodeIndex, nextLayer);
      });
    }
  }

  /**
   * @description
   * Update biases of all nodes in the net, based on the gradient descent update rule
   * (see 'updateBias' function doc in neuron.js)
   */
  function updateBiases(layers, learningRate) {
    for (let i = 1; i < layers.length; i++) {
      layers[i].nodes.forEach(function(node) {
        node.updateBias(learningRate);
      });
    }
  }

  /**
   * @description
   * Update weights of all nodes in the net, based on the gradient descnt update rule
   * (see 'updateWeights' function doc in neuron.js)
   */
  function updateWeights(layers, learningRate) {
    for (let i = 1; i < layers.length; i++) {
      layers[i].nodes.forEach(function(node) {
        node.updateWeights(learningRate, layers[i - 1]);
        node.initialiseError(); // reset the error to 0
      });
    }
  }

  module.exports = NeuralNet;
})();
