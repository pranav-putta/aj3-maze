import abc

import torch
from einops import rearrange

from mazelens.util.structs import ExperienceDict
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, unpad_sequence, unpack_sequence


class Net(torch.nn.Module, abc.ABC):
    """ Abstract class for all policies. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, x: ExperienceDict):
        """ Forward pass of the network. """
        raise NotImplementedError


class StatefulNet(Net, abc.ABC):

    @abc.abstractmethod
    def initialize_hidden(self, batch_size):
        raise NotImplementedError

    @abc.abstractmethod
    def seq_forward(self, x: torch.Tensor, hx, done_mask):
        """
        Forward pass of the network for a sequence of inputs.

        @param x: tensor of shape (B, T, ...)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def single_forward(self, x: torch.Tensor, hx, done_mask):
        """
        Forward pass of the network for a single input.

        @param x: tensor of shape (B, ...)
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, hx: torch.Tensor = None, done_mask: torch.Tensor = None):
        """
        x.states of shape (B, T, ...) or (B, ...)
        @param x:
        @return:
        """
        assert hx is not None, 'hx must be provided for expected behavior'
        assert done_mask is not None, 'done_mask must be provided for expected behavior'

        if len(done_mask.shape) == 1:
            return self.single_forward(x, hx, done_mask)
        else:
            return self.seq_forward(x, hx, done_mask)

    def construct_padded_sequence(self, x, done_mask, hx=None):
        """
        Builds a packed sequence from the input.

        @param x: tensor of shape (B, T, ...)
        @param hx: tensor of shape (B, T, ...)
        @param done_mask: tensor of shape (B, T)
        @return:
        """
        B, T = x.shape[:2]
        x = rearrange(x, 'b t ... -> (b t) ...')
        done_mask = rearrange(done_mask, 'b t -> (b t)')

        episode_start_ids = torch.where(done_mask)[0]
        lengths = (episode_start_ids[1:] - episode_start_ids[:-1]).tolist() + [B * T - episode_start_ids[-1].item()]
        episodes = torch.split(x, lengths)

        padded_x = pad_sequence(episodes, batch_first=True)

        # sort descending by length to pack sequence
        # lengths, perm_idx = torch.tensor(lengths, device='cpu').sort(0, descending=True)
        # padded_x = padded_x[perm_idx]

        if hx is not None:
            hx = rearrange(hx, 'b t ... -> (b t) ...')
            packed_hx = hx[episode_start_ids]
            # packed_hx = packed_hx[perm_idx]
            packed_hx = packed_hx.contiguous()
            return padded_x, packed_hx, lengths
        else:
            return padded_x, lengths

    def deconstruct_packed_sequence(self, packed_x, packed_hx, batch_size):
        """
        Builds a packed sequence from the input.

        @param packed_x: tensor of shape (B, T, ...)
        @param packed_hx: tensor of shape (B, T, ...)
        @return:
        """
        x, lengths = pad_packed_sequence(packed_x, batch_first=True)
        x = unpad_sequence(x, lengths, batch_first=True)
        x = torch.cat(x, dim=0)
        x = rearrange(x, '(b t) ... -> b t ...', b=batch_size)

        # todo; implement unpacking the hidden state as well in the future

        return x, None
