(function() {
  'use strict';

  const sigmoid = require('./utils/sigmoid');
  const sigmoidPrime = require('./utils/sigmoid-prime');


  function Neuron() {
    this.weights = null,
    this.bias = null,
    this.weightedInput = null,
    this.error = null;
    this.gradientsW = null;
    this.gradientB = null;
    this.output = null;
  }


  /**
   * @description
   * Initialise the weights of the neuron. 
   * 
   * The weights of a neuron are just random numbers between 0 and 1. Each weight 
   * is associated to an input in the learning algorithm, which is why the number
   * of weights corresponding to a neuron equals the number of its inputs.
   *    
   * @param {Number} numInputs The number of inputs of the neuron
   */
  Neuron.prototype.initialiseWeights = function(numInputs) {
    this.weights = [];
    for(let i=0; i<numInputs; i++) {
      this.weights.push(Math.random());
    }
  };

  /**
   * @description
   * Initialise the bias of the neuron
   * 
   * The bias is just a random number between 0 and 1
   */
  Neuron.prototype.initialiseBias = function() {
    this.bias = Math.random();
  };

  /**
   * @description
   * Initialise the error in the neuron 
   * 
   * In backpropagation, the error (or delta in academic literature) in a neuron
   * is used to compute the gradient (change) of the cost function with respect 
   * to that neuron's weight/bias 
   */
  Neuron.prototype.initialiseError = function() {
    this.error = 0;
  };

  /**
   * @description
   * Compute the weighted input of a neuron.
   * 
   * The weighted input of a neuron n is the sum of products of each of the neuron's
   * inputs and its corresponding weight added with the bias of the neuron.
   * 
   * For example, for a neuron with inputs x1, x2, x3 and their corresponding 
   * weights w1, w2, w3 and with bias b, the weighted input will be:
   * 
   * WI(n) = (x1*w1 + x2*w2 + x3*w3) + b
   * 
   * It is good to note here, that for neurons in the hidden and output layers
   * the inputs are represented by the output of each neuron in the previous layer
   * (see http://i.stack.imgur.com/76Kuo.png for a good visualization of this concept)
   * 
   * @param {Object} prevLayer The previous layer
   */
  Neuron.prototype.updateWeightedInput = function(prevLayer) {
    this.weightedInput = this.bias;
    prevLayer.nodes.forEach((prevNode, index) => 
      this.weightedInput += prevNode.output * this.weights[index]
    );
  };

  /**
   * @description
   * Compute the output of a neuron
   * 
   * The output of a neuron is computed by applying the cost function to the
   * weighted input of the neuron.
   * 
   * Currently, only the sigmoid function is supported as an activation function of a
   * neuron. More soon :)
   */
  Neuron.prototype.updateOutput = function() {
    this.output = sigmoid(this.weightedInput);
  };

  /**
   * @description
   * Set the error for a neuron in the output layer
   * 
   * This function is based on the first equation of the backpropagation algorithm 
   * (see http://neuralnetworksanddeeplearning.com/chap2.html for details on the
   * four backpropagation equations) and should only be applied for neurons in the
   * output layer
   * 
   * @param {Number} expectedOutput The expectedOutput
   */
  Neuron.prototype.updateOutputError = function(expectedOutput) {
    this.error = (this.output - expectedOutput) * sigmoidPrime(this.weightedInput);
  };

  /**
   * @description
   * Set the error for a neuron in the hidden layer
   * 
   * This function is based on the second equation of the backpropagation algorithm
   * (see http://neuralnetworksanddeeplearning.com/chap2.html for details on the 
   * four backpropagation equations) and should only be applied for neurons in the
   * hidden layers.
   * 
   * @param {Number} crrNodeIndex The index of the current neuron
   * @param {Object} nextLayer  The next layer
   */
  Neuron.prototype.updateError = function(crrNodeIndex, nextLayer) {
    nextLayer.nodes.forEach((nextNode) =>
      this.error += nextNode.error * nextNode.weights[crrNodeIndex]
    );
    this.error *= sigmoidPrime(this.weightedInput);
  }

  /**
   * @description
   * Update the bias of a neuron using the gradient descent update rule in terms
   * of the bias component. (see https://en.wikipedia.org/wiki/Gradient_descent 
   * for details about gradient descent in general and 
   * http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent
   * for gradient descent applied in NN)
   * 
   * @param {Number} learnignRate The learning rate
   *  
   */
  Neuron.prototype.updateBias = function(learningRate) {
    this.bias -= learningRate * this.error; // / batchSize;??
  }

  /**
   * @description
   * Update the weights of a neuron using the gradient descent update rule in terms 
   * of the weight component. (see https://en.wikipedia.org/wiki/Gradient_descent 
   * for details about gradient descent in general and 
   * http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent
   * for gradient descent applied in NN)
   * 
   * @param {Number} learnignRate The learning rate
   * @param {Object} prevLayer The previous layer
   */
  Neuron.prototype.updateWeights = function(learningRate, prevLayer) {
    this.gradientsW = [];
    prevLayer.nodes.forEach((prevNode, index) => {
      let crrGradient = prevNode.output * this.error;
      this.gradientsW.push(crrGradient);
      this.weights[index] -= learningRate * crrGradient;
    });
  }

  module.exports = Neuron;

})();