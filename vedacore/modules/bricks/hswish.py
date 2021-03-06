import torch.nn as nn

from vedacore.misc import registry


@registry.register_module('activation')
class HSwish(nn.Module):
    """Hard Swish Module. Apply the hard swish function:

    Hswish(x) = x * ReLU6(x + 3) / 6

    Args:
        inplace (bool): can optionally do the operation in-place.
            Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.act = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.act(x + 3) / 6
