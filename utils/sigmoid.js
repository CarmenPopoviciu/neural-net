(function() {
  'use strict';

  /**
   * @description
   * The sigmoid activation function (or Standard Logistic Function) 
   * (see https://en.wikipedia.org/wiki/Sigmoid_function)
   * 
   * @param {Number} z The number for which to compute the sigmoid function 
   * @return {Number}
   */
  module.exports = function sigmoid(z) {
    if(!z || (typeof z !== 'number')) {
      throw new Error(`Sigmoid: Can't compute the sigmoidFn for non numerical values, Got ${z}`);
    }
    return (1 / (1 + Math.pow(Math.E, -z)));
  }
  
})();