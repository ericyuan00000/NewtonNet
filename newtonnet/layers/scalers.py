import torch
from torch import nn


# def get_scaler_by_string(key):
#     if key == 'energy':
#         scaler = ScaleShift(scale=True, shift=True)
#     elif key == 'force':
#         scaler = ScaleShift(scale=True, shift=False)
#     elif key == 'hessian':
#         scaler = NullScaleShift()
#     else:
#         print(f'scaler {key} is not supported, use NullScaleShift.')
#         scaler = NullScaleShift()
#     return scaler


class ScaleShift(nn.Module):
    '''
    Node-level scale and shift layer.
    
    Parameters:
        scale (bool): Whether to scale the input.
        shift (bool): Whether to shift the input.
    '''
    def __init__(self, scale=True, shift=True):
        super(ScaleShift, self).__init__()
        # self.z_max = 0
        # if shift is None:
        #     self.shift = None
        # elif shift.numel() == 1:
        #     self.shift = nn.Parameter(shift, requires_grad=True)
        # else:
        #     self.shift = nn.Embedding.from_pretrained(shift.reshape(-1, 1), freeze=False)
        #     self.z_max = max(self.z_max, shift.size(0) - 1)
        # if scale is None:
        #     self.scale = None
        # elif scale.numel() == 1:
        #     self.scale = nn.Parameter(scale, requires_grad=True)
        # else:
        #     self.scale = nn.Embedding.from_pretrained(scale.reshape(-1, 1), freeze=False)
        #     self.z_max = max(self.z_max, scale.size(0) - 1)
        self.scale = nn.Embedding.from_pretrained(torch.ones(118 + 1, 1), freeze=False) if scale else None
        self.shift = nn.Embedding.from_pretrained(torch.zeros(118 + 1, 1), freeze=False) if shift else None

    def forward(self, input, z):
        '''
        Scale and shift input.

        Args:
            input (torch.Tensor): The input values.
            z (torch.Tensor): The atomic numbers of the atoms in the molecule.

        Returns:
            torch.Tensor: The normalized inputs.
        '''
        output = input
        if self.scale is not None:
            output = output * self.scale(z)
        if self.shift is not None:
            output = output + self.shift(z)
        return output
    
    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale is not None}, shift={self.shift is not None})'
    

# class NullScaleShift(nn.Module):
#     '''
#     Null scale and shift layer for untrained properties. Identity function.
#     '''
#     def __init__(self):
#         super(NullScaleShift, self).__init__()
#         self.z_max = 0

#     def forward(self, inputs, z):
#         return inputs