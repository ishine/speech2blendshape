import torch.nn as nn
from src.models.full_deepspeech import MaskConv

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        out = self.act(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, block_nums) -> None:
        super().__init__()

        self.stem = MaskConv(nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=7,
                      stride=(4, 1),
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        ))

        self.in_channels = 32
        self.stage1 = self._make_stage(32, block, block_nums[0], (2, 1))
        self.stage2 = self._make_stage(32, block, block_nums[1], (2, 1))
        self.stage3 = self._make_stage(64, block, block_nums[2], (2, 1))
        self.stage4 = self._make_stage(64, block, block_nums[3], (2, 1))

    def _make_stage(self, out_channels, block, block_num, stride):
        stage = []
        for i in range(block_num):
            if i >= 1:
                stride = 1
            stage.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return MaskConv(nn.Sequential(*stage))

    def forward(self, x, x_length):
        out, x_length = self.stem(x, x_length)
        out, x_length = self.stage1(out, x_length)
        out, x_length = self.stage2(out, x_length)
        out, x_length = self.stage3(out, x_length)
        out, x_length = self.stage4(out, x_length)
        return out

def resnet34():
    return ResNet(BasicBlock, [3, 3, 9, 3])