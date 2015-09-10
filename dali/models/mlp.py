import dali.core as D

class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]

        self.input_layer = D.StackedInputLayer(input_sizes, hiddens[0])
        self.layers = [D.Layer(h_from, h_to) for h_from, h_to in zip(hiddens[:-1], hiddens[1:])]

    def activate(self, inputs):
        assert len(self.layers) == len(self.layer_nonlinearities)
        hidden = self.input_nonlinearity(self.input_layer.activate(inputs))
        for l, nonlinearity in zip(self.layers, self.layer_nonlinearities):
            hidden = nonlinearity(l.activate(hidden))
        return hidden

    def parameters(self):
        ret = self.input_layer.parameters()
        for l in self.layers:
            ret.extend(l.parameters())
        return ret

    def name_parameters(self, prefix):
        self.input_layer.name_parameters(prefix + "_input_layer")
        for layer_idx, layer in enumerate(self.layers):
            layer.name_parameters(prefix + '_layer%d' % (layer_idx,))
