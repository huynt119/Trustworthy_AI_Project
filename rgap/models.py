"""Victim model for the R-GAP demo: same conv topology (kernel/stride/padding,
channel counts) as the LeNet used in ``part1_dlg_lenet`` for DLG/iDLG, so the
two attacks can be compared on "the same LeNet". Two things differ, both
required by the R-GAP linear system (see the reference repo's own CNN6/CNN6d
architectures, ``models/CNN6.py``) and called out in `part3.ipynb`:

  - every layer has ``bias=False`` (R-GAP's per-layer equations, as
    implemented in `recursive_attack.py`, assume ``out = W @ x`` with no
    additive bias term).
  - hidden activation is ``LeakyReLU`` instead of Sigmoid, and the head is a
    single logit (``Linear(768, 1, bias=False)``) trained with binary
    logistic loss instead of 10-way softmax cross-entropy — R-GAP's
    closed-form output-layer inversion (`recursive_attack.inverse_udldu`) is
    only implemented, in the reference code, for a scalar sigmoid output.

The module layout (``self.body`` as a ``ModuleList`` of
``Sequential(OrderedDict([('layer', ...), ('act', ...)]))`` blocks, and a
``forward`` that records each block's input shape) mirrors the reference's
``models/CNN6.py`` exactly, since `part3.ipynb`'s attack loop walks
``net.body`` the same way the reference's ``main.py`` does.
"""
from collections import OrderedDict

import torch.nn as nn


class LeNetRGAP(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU(negative_slope=0.2)
        self.body = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2, bias=False)),
                ('act', act),
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2, bias=False)),
                ('act', act),
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1, bias=False)),
                ('act', act),
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1, bias=False)),
                ('act', act),
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(768, 1, bias=False)),
                ('act', nn.Identity()),
            ])),
        ])

    def forward(self, x):
        x_shape = []
        for block in self.body:
            if isinstance(block.layer, nn.Linear):
                x = x.flatten(1)
            x_shape.append(x.shape)
            x = block(x)
        return x, x_shape

    @staticmethod
    def name():
        return 'LeNetRGAP'


class FCN3(nn.Module):
    """Port of the reference repo's ``models/FCN3.py``: three bias-free
    Linear layers (1000, 100, 1 units), LeakyReLU on every layer including
    the last (unlike LeNetRGAP, which uses Identity on its head). This is
    the architecture the reference README exercises at batch size > 1
    (``python main.py -m FCN3 -d MNIST -b 5``), used here (`part3_batch2.ipynb`)
    as the direct point of comparison against LeNetRGAP's batch=2 behaviour.

    One deliberate deviation from the upstream file: the first layer's input
    width is a constructor argument (``input_dim``) instead of hardcoded
    ``784`` (= 28*28*1, flattened MNIST grayscale). `part3_batch2.ipynb`
    reuses LeNetRGAP's own CIFAR-10 RGB images for this comparison, so it
    passes ``input_dim=3*28*28`` to keep them 3-channel instead of converting
    to grayscale; every other dimension (1000, 100, 1) and every other
    property (bias-free, LeakyReLU throughout) is unchanged from upstream.
    """

    def __init__(self, input_dim=784):
        super().__init__()
        act = nn.LeakyReLU(negative_slope=0.2)
        self.body = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(input_dim, 1000, bias=False)),
                ('act', act),
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(1000, 100, bias=False)),
                ('act', act),
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(100, 1, bias=False)),
                ('act', act),
            ])),
        ])

    def forward(self, x):
        x_shape = []
        for block in self.body:
            if isinstance(block.layer, nn.Linear):
                x = x.flatten(1)
            x_shape.append(x.shape)
            x = block(x)
        return x, x_shape

    @staticmethod
    def name():
        return 'FCN3'


def weights_init(m):
    """Same Uniform(-0.5, 0.5) weight init as part1's LeNet, so the victim
    model sits at the same "random / early FL round" threat model. There is
    no bias to initialize here (all layers are bias-free, see module
    docstring)."""
    if hasattr(m, 'weight') and m.weight is not None:
        m.weight.data.uniform_(-0.5, 0.5)
