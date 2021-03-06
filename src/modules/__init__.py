"""PyTorch Module and ModuleGenerator."""

from src.modules.base_generator import GeneratorAbstract, ModuleGenerator
from src.modules.bottleneck import Bottleneck, BottleneckGenerator
from src.modules.conv import Conv, ConvGenerator, FixedConvGenerator
from src.modules.dwconv import DWConv, DWConvGenerator
from src.modules.flatten import FlattenGenerator
from src.modules.invertedresidualv2 import (InvertedResidualv2,
                                            InvertedResidualv2Generator)
from src.modules.invertedresidualv3 import (InvertedResidualv3,
                                            InvertedResidualv3Generator)
from src.modules.mbconv import MBConv,MBConvGenerator
from src.modules.linear import Linear, LinearGenerator
from src.modules.poolings import (AvgPoolGenerator, GlobalAvgPool,
                                  GlobalAvgPoolGenerator, MaxPoolGenerator)
from src.modules.vit_mobile import MobileViTBlock, MobileViTBlockGenerator, MV2Block, MV2BlockGenerator

from src.modules.convresidual import ConvResidual, ConvResidualGenerator
from src.modules.dropout import DropoutGenerator

__all__ = [
    "ModuleGenerator",
    "GeneratorAbstract",
    "Bottleneck",
    "Conv",
    "ConvResidual",
    "DWConv",
    "Linear",
    "GlobalAvgPool",
    "InvertedResidualv2",
    "InvertedResidualv3",
    "MBConv",
    "BottleneckGenerator",
    "FixedConvGenerator",
    "ConvGenerator",
    "ConvResidualGenerator",
    "LinearGenerator",
    "DWConvGenerator",
    "FlattenGenerator",
    "MaxPoolGenerator",
    "AvgPoolGenerator",
    "GlobalAvgPoolGenerator",
    "InvertedResidualv2Generator",
    "InvertedResidualv3Generator",
    "MBConvGenerator",
    "MobileViTBlock",
    "MobileViTBlockGenerator",
    "MV2Block",
    "MV2BlockGenerator",
    "DropoutGenerator"
]
