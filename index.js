const repl = require('repl')
const replServer = repl.start({
  prompt: 'neuralNet > '
})
replServer.context.NeuralNet = require('./neuralNet')
