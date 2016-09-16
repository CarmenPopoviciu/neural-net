(function() {
  'use strict';

  /**
   * @description
   * The sigmoid activation function (or Standard Logistic Function) 
   * (see https://en.wikipedia.org/wiki/Sigmoid_function)
   * 
   * If the given parameter is a matrix, then the function will apply the sigmoid
   * function for each element and return the resulting array.
   * 
   * @param {Number|Array} z The number or array of numbers for which to compute the sigmoid function 
   * @return {Number|Array}
   */
  module.exports = function sigmoid(z) {
    if(Array.isArray(z)) return sigmoidMatrix(z);
    return sigmoidSimple(z);
  }

  function sigmoidSimple(z) {
    if(!z || (typeof z !== 'number')) {
      throw new Error(`Sigmoid: Can't compute the sigmoidF for non numerical values, Got ${z}`);
    }
    return (1 / (1 + Math.pow(Math.E, -z)));
  }

  function sigmoidMatrix(z) {
    let sigmoid = [];

    z.forEach(function(val) {
      if(Array.isArray(val)) {
        sigmoid.push(sigmoidMatrix(val));
      } else {
        sigmoid.push(sigmoidSimple(val));
      }
    });

    return sigmoid;
  }
  
})();