"""Small helpers ported from the official R-GAP repo's ``utils/utils.py``
(https://github.com/JunyiZhu-AI/R-GAP): activation inverse/derivative pairs
the recursive attack needs to peel each layer, plus the binary logistic loss
used by the victim model's output layer.

Only the pieces this project's LeNet-RGAP variant actually uses are kept
(LeakyReLU + Identity); ``inverse_sigmoid``/``derive_sigmoid`` are kept too
since the reference implements them generically for any activation choice.
"""
import numpy as np
import torch
import torch.nn as nn


def logistic_loss(target, inputs):
    """Binary cross-entropy on a raw logit, mean-reduced (matches the
    reference's ``utils/utils.py::logistic_loss``: ``target`` is the label,
    ``inputs`` is the pre-sigmoid logit produced by the network).
    """
    m = nn.Sigmoid()
    pred = torch.squeeze(m(inputs), -1)
    return torch.mean(-(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)))


def inverse_leakyrelu(x, slope):
    return np.array([a / slope if a < 0 else a for a in x]).astype('float64')


def derive_leakyrelu(x, slope):
    return np.array([slope if a < 0 else 1 for a in x]).reshape(1, -1).astype('float64')


def inverse_identity(x):
    return x


def derive_identity(x):
    return np.ones(x.shape).reshape(1, -1).astype('float64')


def inverse_sigmoid(x):
    return np.array([-np.log(1 / a - 1) for a in x]).astype('float64')


def derive_sigmoid(x):
    return np.array([a * (1 - a) for a in x]).reshape(1, -1).astype('float64')
