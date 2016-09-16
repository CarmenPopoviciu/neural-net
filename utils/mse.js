(function() {
  'use strict';

  /**
   * Mean Squarred Error Cost Function
   * (see https://en.wikipedia.org/wiki/Mean_squared_error)
   */
  module.exports = function mse(expectedOutput, predictedOutput) {
    if(expectedOutput.length !== predictedOutput.length) {
      throw new Error(`MSE: Predicted/expected output vector lengths don\'t match. 
            Got ${expectedOutput.length} for expected output, 
            and ${predictedOutput.length} for predicted output`);
    }

    let mse = 0;
    expectedOutput.forEach(function(value, index) {
      mse += Math.pow((value - predictedOutput[index]), 2); 
    });
    return (mse * 0.5);
  }

})();