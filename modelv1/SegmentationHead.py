from torch import nn, cat

class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(), # why relu? Who knows
            nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x