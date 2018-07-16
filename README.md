# diabetesNeuralNetwork
A neural network using Keras that classifies risk for diabetes based on health data.

# Comparion of models over 10 runs
tensorboard --logdir=struct:Graph-struct/combined/,nostruct:Graph-nostruct/combined --port=7000

struct: input data as it relates to itself is included in input [65,]
nostruct: purely input data [9,]
