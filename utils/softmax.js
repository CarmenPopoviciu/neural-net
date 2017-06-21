(function() {
  'use strict';
  module.exports = function softmax(arr) {
    return arr.map(function(value, index) {
      return (
        Math.exp(value) /
        arr
          .map(function(y /*value*/) {
            return Math.exp(y);
          })
          .reduce(function(a, b) {
            return a + b;
          })
      );
    });
  };
})();
