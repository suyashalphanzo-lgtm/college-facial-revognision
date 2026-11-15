"""The network of our own SFace: MobileFaceNet.

It turns one aligned 112x112 face into 128 numbers (the embedding), the same job OpenCV's SFace does.
MobileFaceNet (Chen et al., 2018) is a small network made for phones, with about 1 million weights.
It is the same network as in the official SFace code (github.com/zhongyy/SFace,
SFace_torch/backbone/model_mobilefacenet.py), written out here with explanations.

Input:  a batch of faces, shape (batch, 3, 112, 112), RGB order, pixel values 0 to 255.
        That is exactly what the app gives it (run_network in app/face_engine.py only swaps
        OpenCV's BGR colour order to RGB), the same input OpenCV's own SFace model takes.
Output: shape (batch, 128). Not yet scaled to length 1; the app does that.

How the picture shrinks on its way through:
    112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7 -> 1x1 (512 numbers) -> 128 numbers
"""

import torch
from torch import nn

EMBEDDING_SIZE = 128


class ConvBlock(nn.Sequential):
    """Convolution, then batch normalization, then PReLU (a ReLU that learns its slope for negative numbers).

    groups=channels makes it a "depthwise" convolution: each channel is filtered on its own,
    which is much cheaper than mixing all channels at once.
    """

    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0, groups=1, activation=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
        ]
        if activation:
            layers.append(nn.PReLU(out_channels))
        super().__init__(*layers)


class Bottleneck(nn.Module):
    """MobileNetV2's building block: widen with a 1x1 convolution, filter each channel with a
    depthwise 3x3 convolution, then narrow again with a 1x1 convolution (no activation).

    With residual=True the block's input is added to its output, so the block only has to learn
    a correction. That makes deep networks much easier to train.
    """

    def __init__(self, in_channels, out_channels, wide_channels, stride=1, residual=False):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_channels, wide_channels),
            ConvBlock(wide_channels, wide_channels, kernel=3, stride=stride, padding=1, groups=wide_channels),
            ConvBlock(wide_channels, out_channels, activation=False),
        )
        self.residual = residual

    def forward(self, x):
        return x + self.layers(x) if self.residual else self.layers(x)


def repeated(count, channels, wide_channels):
    return nn.Sequential(*[Bottleneck(channels, channels, wide_channels, residual=True) for _ in range(count)])


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64, kernel=3, stride=2, padding=1),  # 56x56
            ConvBlock(64, 64, kernel=3, padding=1, groups=64),
            Bottleneck(64, 64, 128, stride=2),  # 28x28
            repeated(4, 64, 128),
            Bottleneck(64, 128, 256, stride=2),  # 14x14
            repeated(6, 128, 256),
            Bottleneck(128, 128, 512, stride=2),  # 7x7
            repeated(2, 128, 256),
            ConvBlock(128, 512),
            # "Global depthwise convolution": one 7x7 filter per channel turns the 7x7 map into 1 number.
            # Unlike a plain average, it learns that some parts of the face (eyes, nose) matter more.
            ConvBlock(512, 512, kernel=7, groups=512, activation=False),  # 1x1
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size, eps=2e-5),
        )

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        # Pixels 0..255 become roughly -1..1, which networks train well on. Doing it here, inside the
        # model, means the exported ONNX file accepts the raw pixels OpenCV gives it.
        x = (faces - 127.5) / 128.0
        return self.embedding(self.features(x))
