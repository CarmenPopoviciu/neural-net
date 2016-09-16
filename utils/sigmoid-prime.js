(function() {
  'use strict';

  var sigmoid = require('./sigmoid');

   /**
   * The derivative of the sigmoid function
   * @param {Number|Array} z
   * @return {Number|Array}
   */
  module.exports = function sigmoidPrime(z) {
    if(Array.isArray(z)) return sigmoidPrimeMatrix(z);
    return sigmoidPrimeSimple(z);
  }

  /////

  function sigmoidPrimeSimple(z) {
    return sigmoid(z) * (1 - sigmoid(z));
  }

  function sigmoidPrimeMatrix(z) {
    let sigmoidPrime = [];

    z.forEach(function(val) {
      if(Array.isArray(val)) {
        sigmoidPrime.push(sigmoidPrimeMatrix(val));
      } else {
        sigmoidPrime.push(sigmoidPrimeSimple(val));
      }
    });

    return sigmoidPrime;
  }
  
})();