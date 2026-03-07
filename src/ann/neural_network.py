from .neural_layer import NeuralLayer

class NeuralNetwork:
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 num_layers,
                 output_size,
                 activation="relu",
                 weight_init="random"):

        self.layers = []

        
        # (Direct connection from input to output)
        if num_layers == 0:
            self.layers.append(
                NeuralLayer(
                    input_size=input_size,
                    output_size=output_size,
                    activation="softmax",
                    weight_init=weight_init
                )
            )
        else:

            # First hidden layer

            self.layers.append(
                NeuralLayer(
                    input_size=input_size,
                    output_size=hidden_sizes[0],
                    activation=activation,
                    weight_init=weight_init
                )
            )

            # Remaining hidden layers
            for i in range(1, num_layers):
                self.layers.append(
                    NeuralLayer(
                        input_size=hidden_sizes[i-1],
                        output_size=hidden_sizes[i],
                        activation=activation,
                        weight_init=weight_init
                    )
                )

            # Output layer 
            # last hidden to output classes
            # activation - softmax
            self.layers.append(
                NeuralLayer(
                    input_size=hidden_sizes[-1],
                    output_size=output_size,
                    activation="softmax",
                    weight_init=weight_init
                )
            )

    # FORWARD PASS
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out


    # BACKWARD PASS

    def backward(self, dA):
        grad = dA
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        return grad 

    def get_layers(self):
        return self.layers