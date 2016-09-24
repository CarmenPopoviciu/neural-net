(function() {
  'use strict';

  const sigmoid = require('./utils/sigmoid');
  const sigmoidPrime = require('./utils/sigmoid-prime');

  const activations = ['SIGMOID'];
  
  module.exports = {
    /**
     * @description
     * The sigmoid activation. 
     */
    SIGMOID: {
      fn: sigmoid,
      fnPrime: sigmoidPrime
    },
    /**
     * @description
     * Return the list of supported activations
     * 
     * @return {Array}
     */
    activations: function() {
      return activations;
    },
    /**
     * @description
     * Check if given activation is supported
     * 
     * @param {String} activation The activation name (eg 'SIGMOID')
     * @return {Boolean} 
     */
    isSupportedActivation: function(activation) {
      activation = activation.toUpperCase();
      return activations.includes(activation);
    }
  }

})();