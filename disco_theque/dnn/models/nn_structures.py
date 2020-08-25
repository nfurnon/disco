import torch
import numpy as np
import torch.nn as nn
# %%


class CustomModule(nn.Module):
    """
    Custom module to add the function multiply_argument_to_list
    """
    def __init__(self):
        super(CustomModule, self).__init__()

    def multiply_argument_to_list(self, arg, class_attr_name, n):
        """
        If arg is a single element, turn it into a n-list of the repeated element.
        This argument name is then attributed to the class under the name class_attr_name.
        Args:
            arg (float, list, tuple):   Argument 
            class_attr_name (string):   Name of the attribute.
            n (int):                    Number of repetitions
        Returns:
            nothing, but the class parent class has an additional or changed attribute `class_attr_name`
        """

        if type(arg) not in (list, tuple):      # Assumed is then that it is a scalar
            arg = [arg]
            setattr(self, class_attr_name, arg * n)
        elif type(arg) == tuple:                # Assumed is that we stack `n` times `arg`.
            arg = [arg for _ in range(n)]
            setattr(self, class_attr_name, arg)
        elif type(arg) == list:                 # Assumed is that arg is already correct (len `n`).
            setattr(self, class_attr_name, arg)
        else:
            raise ValueError("type of `arg` should be scalar, tuple or list. Got {}".format(type(arg)))


# %%
class FF(CustomModule):
    """
    Feed-forward class of len(nb_units) layers, where activation functions are given by `activations`.
    """
    def __init__(self, input_size, nb_units, activations):
        """
        Args:
            input_size (int):       input number of features
            nb_units (list, tuple): nb of features in each layer
            activations (str):      name of the activation (e.g. 'ReLU', 'Tanh')
        """
        super(FF, self).__init__()
        self.input_size = input_size
        if type(nb_units) not in (tuple, list):
            self.nb_units = [nb_units]
        else:
            self.nb_units = nb_units
        self.n_layers = len(self.nb_units)
        self.multiply_argument_to_list(activations, 'activations', self.n_layers)

        self.layers = self.build_structure()

    def build_structure(self):
        layers = nn.ModuleList()
        for i_layer in range(self.n_layers):
            if i_layer == 0:
                in_features = self.input_size
            else:
                in_features = self.nb_units[i_layer - 1]
            out_features = self.nb_units[i_layer]
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))

        return layers

    def forward(self, x):
        for i_layer, layer in enumerate(self.layers):
            x = getattr(torch, self.activations[i_layer])(layer(x))
        return x


# %%
class RNNSingle(nn.Module):
    """
    Class of a single RNN layer, which outputs in the forward() definition only the layer output,
    and not the hidden states.
    This is to be used in deep RNN, where the layers are stacked using Sequential()
    """
    def __init__(self, cell_type, *args, **kwargs):
        super(RNNSingle, self).__init__()
        layer_cell = getattr(nn, str.upper(cell_type))
        self.rnn_layer = layer_cell(*args, **kwargs)

    def forward(self, x):
        x, _ = self.rnn_layer(x)
        return x


class RNN(CustomModule):
    """
    RNN somehow flexible brick. Self-sufficient but can be included in a C-RNN.
    """
    def __init__(self, input_size, hidden_layers_units, cell_architecture,
                 batch_first=True, dropouts=0, bidirectional_layers=False):
        """
        Args:
            input_size (int): Input features
            hidden_layers_unit (list[int]): List of number of features in the hidden layers
            cell_architecture (str): type of cell architecture (RNN, LSTM, GRU)
            batch_first (bool): If True, then the input and output tensors are provided as 
                                (batch, seq, feature) [True]
            dropouts (int or list[int]): introduces a Dropout layer on the outputs of each 
                                         RNN layer except the last layer [0]
            bidirectional_layers (bool or list[bool]): Use bidirectional RNNs ? [False]
        """
        super().__init__()      # Initialize parent class

        self.input_size = input_size
        self.hidden_units = hidden_layers_units
        self.n_rnn_layers = len(hidden_layers_units)
        self.cell_archi = cell_architecture
        self.batch_first = batch_first

        # Make dropouts and bidirectional_layers arrays of length the number of layers
        if type(dropouts) not in (list, tuple):
            dropouts = [dropouts]
        if len(dropouts) == 1:
            self.dropouts = dropouts * self.n_rnn_layers
            self.dropouts[-1] = 0
        else:
            assert (len(dropouts) == self.n_rnn_layers), \
                "Please specify only one dropout value or as many as the layers number. " \
                "Last one should be 0"
            self.dropouts = dropouts

        self.multiply_argument_to_list(bidirectional_layers, 'bidirectional_layers', self.n_rnn_layers)

        self.model = nn.Sequential(*self.build_structure())

    def build_structure(self):
        """
        Subfunction to avoid overloading __init__. Associate all the LSTM/GRU blocks into a DNN.
        Returns:
            All the layers in a list
        """
        layers = []
        for i_layer in range(self.n_rnn_layers):
            if i_layer == 0:
                input_size = self.input_size
            else:
                input_size = self.hidden_units[i_layer - 1]

            rnn_layer = RNNSingle(self.cell_archi, input_size=input_size, hidden_size=self.hidden_units[i_layer],
                                  num_layers=1, batch_first=self.batch_first, dropout=self.dropouts[i_layer],
                                  bidirectional=self.bidirectional_layers[i_layer])
            layers.append(rnn_layer)

        return layers

    def forward(self, x):
        return self.model(x)


# %%
class CNN2d(CustomModule):
    """
    2D-Convolutional neural network (kernels are two-dimensional tensors).
    """
    def __init__(self, n_channels, conv_kernels, conv_strides, pool_kernels, pool_strides,
                 conv_padding=0, conv_dilation=1, conv_groups=1, conv_bias=True,
                 pool_types='Max', pool_padding=0, pool_dilation=1):
        """
        Args:
            n_channels (int):    Number of channels at all layers, including the input one.
            conv_kernels (list): Kernel size of all convolutional layers.
            conv_strides (list[tuple]):  Strides in (x, y) directions of convolutional layers.
            pool_kernels (list): Size of the pooling kernel. None at layers where no pooling should be performed.
            pool_strides (list[tuple]):  Strides in (x, y) directions of pooling layers.
            conv_padding (int or tuple[int, int]): implicit zero-paddings before convolutional layer [0]
            conv_dilation (int or tuple[int, int]): spacing between the kernel points [1]
            conv_groups (int): connections between inputs and outputs [1]
            conv_bias (bool): If True, adds a learnable bias to the output [True]
            pool_types (str or list[str]): Type of the pooling layers. None at layers where no pooling should be 
                                           performed ['Max']
            pool_padding (int or tuple[int, int]): implicit zero-paddings before padding layer [0]
            pool_dilation (int or tuple[int, int]): spacing between the pooling points [1]
        """
        super(CNN2d, self).__init__()
        self.n_cnn_layers = len(n_channels) - 1
        self.n_channels = n_channels
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.pool_kernels = pool_kernels
        self.pool_strides = pool_strides
        for arg_name in ['conv_padding', 'conv_dilation', 'conv_groups', 'conv_bias',
                         'pool_types', 'pool_padding', 'pool_dilation']:
            self.multiply_argument_to_list(eval(arg_name), arg_name, self.n_cnn_layers)

        self.model = nn.Sequential(*self.build_structure())

    def build_structure(self):
        layers = []
        for i_layer in range(self.n_cnn_layers):
            conv_layer = nn.Conv2d(in_channels=self.n_channels[i_layer], out_channels=self.n_channels[i_layer + 1],
                                   kernel_size=self.conv_kernels[i_layer], stride=self.conv_strides[i_layer],
                                   padding=self.conv_padding[i_layer], dilation=self.conv_dilation[i_layer],
                                   groups=self.conv_groups[i_layer], bias=self.conv_bias[i_layer])
            norm_layer = nn.BatchNorm2d(self.n_channels[i_layer + 1])
            pool_name = self.pool_types[i_layer] + 'Pool2d'
            pool_layer = getattr(nn, pool_name)(self.pool_kernels[i_layer], stride=self.pool_strides[i_layer],
                                                padding=self.pool_padding[i_layer],
                                                dilation=self.pool_dilation[i_layer])
            layers.append(conv_layer)
            layers.append(norm_layer)
            layers.append(pool_layer)

        return layers

    def forward(self, x):
        return self.model(x)

    def get_output_dim(self, input_shape):
        """Return the output dimension of all CNN layers.
           See https://pytorch.org/docs/stable/nn.html#conv2d and https://pytorch.org/docs/stable/nn.html#maxpoold for
           formulas
        """
        h_in = input_shape[-2]
        w_in = input_shape[-1]
        for i in range(self.n_cnn_layers):
            cs = [self.conv_strides[i]] if type(self.conv_strides[i]) not in (tuple, list) else self.conv_strides[i]
            cp = [self.conv_padding[i]] if type(self.conv_padding[i]) not in (tuple, list) else self.conv_padding[i]
            ck = [self.conv_kernels[i]] if type(self.conv_kernels[i]) not in (tuple, list) else self.conv_kernels[i]
            cd = [self.conv_dilation[i]] if type(self.conv_dilation[i]) not in (tuple, list) else self.conv_dilation[i]
            cs = ck if cs == [None] else cs

            ps = [self.pool_strides[i]] if type(self.pool_strides[i]) not in (tuple, list) else self.pool_strides[i]
            pp = [self.pool_padding[i]] if type(self.pool_padding[i]) not in (tuple, list) else self.pool_padding[i]
            pk = [self.pool_kernels[i]] if type(self.pool_kernels[i]) not in (tuple, list) else self.pool_kernels[i]
            pd = [self.pool_dilation[i]] if type(self.pool_dilation[i]) not in (tuple, list) else self.pool_dilation[i]
            ps = pk if ps == [None] else ps

            # Output of convolution
            h_in = np.floor((h_in + 2 * cp[0] - cd[0] * (ck[0] - 1) - 1) / cs[0] + 1)
            w_in = np.floor((w_in + 2 * cp[-1] - cd[-1] * (ck[-1] - 1) - 1) / cs[-1] + 1)
            # Output of pooling
            h_in = np.floor((h_in + 2 * pp[0] - pd[0] * (pk[0] - 1) - 1) / ps[0] + 1)
            w_in = np.floor((w_in + 2 * pp[-1] - pd[-1] * (pk[-1] - 1) - 1) / ps[-1] + 1)
        return int(h_in), int(w_in)

