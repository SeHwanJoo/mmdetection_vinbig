import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from efficientnet_pytorch import EfficientNet
from ..utils import ResLayer


@BACKBONES.register_module()
class EfficientNetM(nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        super(EfficientNetM, self).__init__()
        assert model_name in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
                              'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
                              'efficientnet-b6', 'efficientnet-b7']
        self.model = EfficientNet.from_pretrained(model_name, include_top=False)
        self.conv_stem = self.model._conv_stem
        self.bn = self.model._bn0
        self.blocks = self.model._blocks
        self.swish = self.model._swish
        if model_name == 'efficientnet-b0':
            self.indices = [2, 4, 10, 15]
        elif model_name == 'efficientnet-b1':
            self.indices = [4, 7, 15, 22]
        elif model_name == 'efficientnet-b2':
            self.indices = [4, 7, 15, 22]
        elif model_name == 'efficientnet-b3':
            self.indices = [4, 7, 17, 25]
        elif model_name == 'efficientnet-b4':
            self.indices = [5, 9, 21, 31]
        elif model_name == 'efficientnet-b5':
            self.indices = [7, 12, 26, 38]
        elif model_name == 'efficientnet-b6':
            self.indices = [8, 14, 30, 44]
        elif model_name == 'efficientnet-b7':
            self.indices = [10, 17, 37, 54]

    def forward(self, x):
        out = []
        x = self.swish(self.bn(self.conv_stem(x)))
        drop_connect_rate = self.model._global_params.drop_connect_rate
        for i, block in enumerate(self.blocks):
            if drop_connect_rate:
                drop_connect_rate *= float(i) / len(self.blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if i in self.indices:
                out.append(x)

        return tuple(out)

    def init_weights(self, pretrained=None):
        print('init_weight')