(function() {
  'use strict';

  var sigmoid = require('./sigmoid');

  /**
   * @description
   * The derivative of the sigmoid function
   * 
   * @param {Number} z The number to compute sigmoid prime for
   * @return {Number}
   */
  module.exports = function sigmoidPrime(z) {
    return sigmoid(z) * (1 - sigmoid(z));
  }
  
})();