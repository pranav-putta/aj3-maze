import torch
from torch.nn import GRU

from mazelens.nets.base_net import StatefulNet


class RNNStateEncoder(StatefulNet):

    def __init__(self, rnn_type, hidden_dim, layers):
        super().__init__()

        if rnn_type == 'gru':
            self.rnn = GRU(hidden_dim, hidden_dim, num_layers=layers, batch_first=True)
        else:
            raise NotImplementedError('Only GRU is supported')

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.layers = layers

    def single_forward(self, x, hx, mask):
        not_done_mask = ~mask

        # mask out previous hidden states accordingly
        # this is to prevent the rnn from using the hidden states of the previous episode
        hx = hx * not_done_mask[None, :, None]

        # run rnn
        x, hx = self.rnn(x, hx)
        return x, hx

    def seq_forward(self, x, hx, done_mask):
        B, *_ = x.shape
        packed_x, packed_hx = self.construct_packed_sequence(x, hx, done_mask)
        packed_out, packed_hx = self.rnn(packed_x, packed_hx)
        x, hx = self.deconstruct_packed_sequence(packed_out, packed_hx, B)
        return x, hx
        # return self.rnn(x)

    def initialize_hidden(self, batch_size):
        return torch.zeros(self.layers, batch_size, self.hidden_dim)
