import torch.nn as nn
import torchvision

from ..builder import BACKBONES
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger


@BACKBONES.register_module()
class DenseNet121(nn.Module):
    def __init__(self, indicies=[4, 6, 8, 10]):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True).features
        self.indicies = indicies

    def forward(self, x):  # should return a tuple
        outs = []
        for i, feature in enumerate(self.model):
            x = feature(x)
            if i in self.indicies:
                outs.append(x)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')
