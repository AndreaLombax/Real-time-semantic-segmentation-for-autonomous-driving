from typing import List
from torch import nn

# A single SegFormerDecoderBlock contains one upsample layer (for the spatial dimension)
#  and one conv layer (for the channels). 
# The scale_factor parameter is needed to tell it how much we want to upsample the feature.
class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

# SegFormerDecoder is just a list of DecoderBlocks.
# It takes a list of features and returns a list of new features with the same spatial size and channels.
class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    
    def forward(self, features):
        new_features = []
        for feature, stage in zip(features,self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features