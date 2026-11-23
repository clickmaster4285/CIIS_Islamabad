"""OSNet x1_0 (omni-scale person re-ID network) — inference only.

Minimal re-implementation of the network from deep-person-reid (Zhou et al., MIT
licence) so the OSNet checkpoint in runs/reid/ loads without installing `torchreid`
(which needs a C compiler on Windows). Layer names match the original exactly, so the
checkpoint loads with strict=True. In eval mode the model returns the 512-d feature.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    """conv + bn + relu (stem)."""

    def __init__(self, in_c: int, out_c: int, k: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 conv + bn, no activation."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """1x1 pointwise + 3x3 depthwise + bn + relu."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False, groups=out_c)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Per-channel attention shared by all scale streams."""

    def __init__(self, in_c: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_c, in_c // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(in_c // reduction, in_c, 1, bias=True)

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, 1)
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(g))))
        return x * g


class OSBlock(nn.Module):
    """Omni-scale block: 4 streams (1-4 stacked 3x3) fused by a shared gate."""

    def __init__(self, in_c: int, out_c: int, bottleneck_reduction: int = 4):
        super().__init__()
        mid = out_c // bottleneck_reduction
        self.conv1 = Conv1x1(in_c, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_c)
        self.downsample = Conv1x1Linear(in_c, out_c) if in_c != out_c else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = (
            self.gate(self.conv2a(x1))
            + self.gate(self.conv2b(x1))
            + self.gate(self.conv2c(x1))
            + self.gate(self.conv2d(x1))
        )
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


def _make_layer(n_blocks: int, in_c: int, out_c: int, reduce_spatial: bool) -> nn.Sequential:
    layers: list[nn.Module] = [OSBlock(in_c, out_c)]
    layers += [OSBlock(out_c, out_c) for _ in range(n_blocks - 1)]
    if reduce_spatial:
        layers.append(nn.Sequential(Conv1x1(out_c, out_c), nn.AvgPool2d(2, stride=2)))
    return nn.Sequential(*layers)


class OSNet(nn.Module):
    def __init__(self, num_classes: int = 751, channels=(64, 256, 384, 512), layers=(2, 2, 2), feature_dim: int = 512):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = _make_layer(layers[0], channels[0], channels[1], reduce_spatial=True)
        self.conv3 = _make_layer(layers[1], channels[1], channels[2], reduce_spatial=True)
        self.conv4 = _make_layer(layers[2], channels[2], channels[3], reduce_spatial=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        v = F.adaptive_avg_pool2d(x, 1).flatten(1)
        v = self.fc(v)
        if not self.training:
            return v
        return self.classifier(v)


def load_checkpoint_state(weights_path: str) -> dict:
    """Load a checkpoint as plain tensors (``weights_only=True``: no arbitrary pickle code).

    Old deep-person-reid checkpoints that also pickle non-tensor objects cannot load that way;
    only then fall back to a full unpickle, with a loud warning (trust only your own weights).
    """
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # UnpicklingError / RuntimeError depending on torch version
        warnings.warn(
            f"[reid] {weights_path} is not a plain tensor checkpoint ({exc}); "
            "loading with weights_only=False — only use trusted weight files",
            RuntimeWarning,
            stacklevel=2,
        )
        print(f"[reid] WARNING: {weights_path} needs weights_only=False to load (trusted file only)")
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def load_osnet_x1_0(weights_path: str, device: str = "cpu") -> OSNet:
    """Build OSNet x1_0 and load a deep-person-reid checkpoint (strict: every tensor must match)."""
    state = load_checkpoint_state(weights_path)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    num_classes = int(state["classifier.weight"].shape[0]) if "classifier.weight" in state else 751
    model = OSNet(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)
