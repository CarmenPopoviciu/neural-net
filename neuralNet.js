(function() {
  'use strict';

  function neuralNet(config) {
    var layers = [];

    if(config && config.layers && config.layers.length) init(config.layers);

    return {
      addLayer: addLayer,
      train: train,
      predict: predict
    };

    /**
     * Initialise net with given layers. 
     * @param {Array} layers The layers in the net 
     * @return void
     */
    function init(layers) {
      layers.forEach(function(numNodes) {
        addLayer(numNodes);
      });
    }

    /** PUBLIC API ****************************************************/

    /**
     * Adds a new layer to the network, composed of numNodes neurons
     * @param {Number} numNodes Number of neurons in the layer
     */
    function addLayer(numNodes) {
      if(!numNodes) throw new Error('Invalid argument: Undefined number of nodes. Please specify the number of neurons in the layer.');
      if(typeof numNodes !== 'number') throw new Error('Invalid argument: Number of nodes should be a number, got ' + typeof numNodes + ' instead.');
      if(!Number.isInteger(numNodes)) throw new Error('Invalid argument: Number of nodes should be an integer.');
      
      if(!layers.length) {
        layers.push({ // first layer is the input layer, so no weights or biases needed
          numNodes: numNodes
        }); 
        console.log('No weights or biases needed for the input layer. Skipping step.');
        console.log("Input layer created!");
      } else {
        layers.push({
          numNodes: numNodes,
          weights: initialiseWeights(numNodes, layers[layers.length-1].numNodes),
          biases: initialiseBiases(numNodes)
        });
        console.log('Layer ' + layers.length + ' created!');
      }
    }

    /**
     * Training algorithm
     * @param {Array} data The training data
     */
    function train(input, output) {
      // TODO(@carmen) validate input and output based on nn layout
      feedforward(input, layers);
    }

    /**
     * The prediction function. Predicts the output for a given input
     * @param {Array} data The data set
     * @return {Array} The predicted output 
     */
    function predict(data) {
    }

    /** end PUBLIC API ****************************************************/

    /**
     * Initialises biases for a given number of neurons
     * @param {Number} numNodes The number of neurons in layer
     * @return {Array} The biases vector
     */
    function initialiseBiases(numNodes) {
      var biases = [];
      for(var i=0; i<numNodes; i++) {
        biases.push(Math.random());
      }
      console.log('Biases initialised to ', biases);
      return biases;
    }

    /**
     * Initialises the weights connecting two consecutive layers
     * @param {Number} numNodesLayer1 The number of nodes in layer i
     * @param {Number} numNodesLayer2 The number of nodes in layer i-1
     * @return {Array} The weights vector
     */
    // TODO(@carmen) param naming is crap
    function initialiseWeights(numNodesLayerTo, numNodesLayerFrom) {
      var weights = [];

      for(var i=0; i<numNodesLayerTo; i++) {
        var crrNodeWeights = [];
        for(var j=0; j<numNodesLayerFrom; j++) {
          crrNodeWeights.push(Math.random());
        }
        weights.push(crrNodeWeights);
      }
      console.log('Weights initialised to ', weights);
      return weights;
    }

    /**
     * Feedforward flow
     * TODO(@carmen) add proper desc 
     * @param {Array} input The training input
     * @param {Array} layers Array of layer objects that describe the topology of the network
     * @return {Array} The network's predicted output 
     */
    function feedforward(input, layers) {
      // TODO(@carmen) validation of args?
      layers.forEach(function(crrLayer, index) {
        crrLayer.activations = [];

        if(index === 0) {
          Object.assign(crrLayer.activations, input); // the activation(output) vector of the input layer is the input vector itself
        } else {
          var prevLayer = layers[index-1],
              crrNodeActivation;

          for(var i=0; i<crrLayer.numNodes; i++) {
            crrNodeActivation = sigmoid(dotProduct(prevLayer.activations, crrLayer.weights[i]) + crrLayer.biases[i]);
            crrLayer.activations.push(crrNodeActivation);
          }
        }
        console.log('Layer ' + (index+1) + '\'s output is: ' + crrLayer.activations);
      });

      console.log('Network\'s predicted output is: ' + layers[layers.length-1].activations);
      return layers[layers.length-1].activations;
    }

    /**
     * The sigmoid activation function (or Standard Logistic Function) (see https://en.wikipedia.org/wiki/Sigmoid_function)
     * TODO(@carmen) give a short desc
     * @param {Number} z 
     * @return {Number}
     */
    function sigmoid(z) {
      return (1 / (1 + Math.pow(Math.E, -z)));
    }

    /**
     * Computes the scalar product of two vectors. 
     * Algebraically, the scalar(or dot) product is the sum of the products of the 
     * corresponding entries of the two sequences of numbers 
     * (see https://en.wikipedia.org/wiki/Dot_product).
     * 
     * For example, given vector A = [a1, a2,..., an] and vector B = [b1, b2,..., bn] 
     * with n representing the dimension of the vector space,
     * the dot product equals AB = a1*b1 + a2*b2 + ... + an*bn
     * 
     * @param {Array} vectorA First vector
     * @param {Array} vectorB Second vector
     * @return {Number} The value of the dot product
     */
    function dotProduct(vectorA, vectorB) {
      if(!vectorA || !vectorB) throw new Error('Invalid argument: Undefined vector(s) in dot product function call.');
      if(vectorA.length !== vectorB.length) throw new Error('Unequal vector sizes: Vectors of a scalar product should have the same size.');

      return vectorA.reduce(function(prevValue, crrValue, crrIndex) {
        return prevValue + (crrValue * vectorB[crrIndex]);
      }, 0);
    }
  }
  
  module.exports = neuralNet;
})();