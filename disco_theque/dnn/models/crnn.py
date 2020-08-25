from disco_theque.dnn.models.nn_structures import CNN2d, RNN, FF
from disco_theque.dnn.utils import get_loss_frames
import numpy as np
from torch import nn
from torch.optim import RMSprop


# %%
class CRNN(nn.Module):
    """ CRNN model"""
    def __init__(self, input_shape, cnn_filters, conv_kernels, conv_strides, pool_kernels, pool_strides,
                 rnn_units, rnn_cell,
                 ff_units,
                 conv_padding=0, conv_dilation=1, conv_groups=1, conv_bias=True,
                 pool_types='Max', pool_padding=0, pool_dilation=1,
                 rnn_dropouts=0, rnn_bi=False,
                 ff_activation='sigmoid'):
        """
        Args:
            input_shape (tuple[int, int, int]): Input shape (n_ch, time_dim, feature_dim)
            cnn_filters (list[int] or tuple[int]): Number of filters in each convolutional layer
            conv_kernels (list): Kernel size of all convolutional layers.
            conv_strides (list[tuple]):  Strides in (x, y) directions of convolutional layers.
            pool_kernels (list): Size of the pooling kernel. None at layers where no pooling should be performed.
            pool_strides (list[tuple]):  Strides in (x, y) directions of pooling layers.
            rnn_units (list[int]): List of number of features in the hidden RNN layers
            rnn_cell (str): type of RNN cell architecture (RNN, LSTM, GRU)
            ff_units (list, tuple): nb of features in each FF layer
            conv_padding (int or tuple[int, int]): implicit zero-paddings before convolutional layer [0]
            conv_dilation (int or tuple[int, int]): spacing between the kernel points [1]
            conv_groups (int): connections between inputs and outputs [1]
            conv_bias (bool): If True, adds a learnable bias to the output [True]
            pool_types (str or list[str]): Type of the pooling layers. None at layers where no pooling should be 
                                           performed ['Max']
            pool_padding (int or tuple[int, int]): implicit zero-paddings before padding layer [0]
            pool_dilation (int or tuple[int, int]): spacing between the pooling points [1]
            rnn_dropouts (int or list[int]): introduces a Dropout layer on the outputs of each 
                                             RNN layer except the last layer [0]
            rnn_bi (bool or list[bool]): Use bidirectional RNNs ? [False]
            activations (str):      name of the activation (e.g. 'ReLU', 'Tanh') ['sigmoid']
        """
        super(CRNN, self).__init__()
        self.input_shape = input_shape
        n_channels = [input_shape[0]] + [*cnn_filters]

        self.cnn = CNN2d(n_channels, conv_kernels, conv_strides, pool_kernels, pool_strides,
                         conv_padding=conv_padding, conv_dilation=conv_dilation, conv_groups=conv_groups,
                         conv_bias=conv_bias, pool_types=pool_types, pool_padding=pool_padding,
                         pool_dilation=pool_dilation)
        self.x_out, y_out = self.cnn.get_output_dim(input_shape)
        self.rnn = RNN(int(n_channels[-1] * y_out), rnn_units, rnn_cell,
                       batch_first=True, dropouts=rnn_dropouts, bidirectional_layers=rnn_bi)
        self.ff = FF(rnn_units[-1], ff_units, ff_activation)

    def forward(self, inp):
        if len(inp.size()) == 3:
            inp = inp.view(inp.size(0), 1, inp.size(1), inp.size(2))
        x = self.cnn(inp)
        x = x.view(x.size(0), x.size(2), x.size(1) * x.size(-1))    # Make it 3D, keeping time dimension constant
        x = self.rnn(x)
        x = self.ff(x.squeeze())

        return x

    def get_loss_frames(self, output_frames):
        """
        Given the NN architecture, return the frames that match between input and output, in order to compute the loss.
        :return:
        """
        win_len_in = self.input_shape[1]
        win_len_out = self.x_out
        if output_frames == 'last':
            new_len = (win_len_in + win_len_out) // 2
            ff_in = new_len - 1
            lf_in = ff_in + 1
        elif output_frames == 'mid':
            ff_in = int(np.ceil(win_len_in) / 2)
            lf_in = ff_in + 1
        elif output_frames == 'all':
            ff_in = (win_len_in - win_len_out) // 2
            lf_in = (win_len_in + win_len_out) // 2
        else:
            raise ValueError("Unknown argument value {}. It should be either 'all', 'mid' or 'last'."
                             .format(output_frames))
        ff_out, lf_out = get_loss_frames(win_len_out, output_frames)

        return (ff_in, lf_in), (ff_out, lf_out)


def build_crnn(input_shape, cnn_filters, conv_kernels, conv_strides, pool_kernels, pool_strides,
               rnn_units, rnn_cell,
               ff_units,
               conv_padding=0, conv_dilation=1, conv_groups=1, conv_bias=True,
               pool_types='Max', pool_padding=0, pool_dilation=1,
               rnn_dropouts=0, rnn_bi=False,
               ff_activation='sigmoid'):
    crnn_model = CRNN(input_shape, cnn_filters, conv_kernels, conv_strides, pool_kernels, pool_strides,
                      rnn_units, rnn_cell,
                      ff_units,
                      conv_padding=conv_padding, conv_dilation=conv_dilation,
                      conv_groups=conv_groups, conv_bias=conv_bias,
                      pool_types=pool_types, pool_padding=pool_padding, pool_dilation=pool_dilation,
                      rnn_dropouts=rnn_dropouts, rnn_bi=rnn_bi,
                      ff_activation=ff_activation)
    crnn_optimizer = RMSprop(crnn_model.parameters(), lr=0.001)
    crnn_optimizer.clip = False

    return crnn_model, crnn_optimizer

