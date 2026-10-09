"""Core R-GAP recursive attack: peel a network's gradient one layer at a
time, from the output back to the input, solving a (small) linear system per
layer instead of running gradient-based optimization.

Ported from the official R-GAP repository
(https://github.com/JunyiZhu-AI/R-GAP, ``recursive_attack.py``), which
implements:

    Junyi Zhu, Matthew B. Blaschko. "R-GAP: Recursive Gradient Attack on
    Privacy." ICLR 2021.

Two changes vs. the reference, both purely compatibility/robustness fixes,
no change in algorithm:
  - ``peeling``: ``np.bool`` (removed in NumPy >= 1.24) -> ``bool``.
  - ``inverse_udldu``: run on the caller's device/dtype instead of a
    module-level hardcoded CPU ``setup``, so it works on CUDA too.
"""
import numpy as np
import torch
import torch.nn as nn

from .conv2circulant import generate_coordinates, circulant_w, aggregate_g


def inverse_udldu(udldu, device='cpu', dtype=torch.float64, iterations=30_000, lr=0.01):
    """Recover the scalar pre-activation ``u`` of a bias-free, single-logit
    output layer trained with binary logistic loss, given only
    ``udldu = dot(dL/dW, W) = dL/du * u``.

    ``dL/du = sigmoid(u) - 1`` for the (fixed, by construction of the attack
    label below) convention used here, so ``udldu = -u / (1 + exp(u))``: a
    1-D transcendental equation in ``u``, solved by gradient descent exactly
    like the reference implementation (there is no closed form).
    """
    setup = dict(device=device, dtype=dtype)
    u = torch.tensor(0.0).to(**setup).requires_grad_(True)
    udldu = torch.tensor(udldu).to(**setup)
    optimizer = torch.optim.Adam([u], lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(iterations):
        optimizer.zero_grad()
        udldu_ = -u / (1 + torch.exp(u))
        l = loss_fn(udldu_, udldu)
        l.backward()
        optimizer.step()
    udldu_ = -u / (1 + torch.exp(u))
    print(f"[R-GAP] inverse_udldu residual: {udldu.item() - udldu_.item():.2e}")
    return u.detach().cpu().numpy()


def peeling(in_shape, padding):
    """Boolean mask, True at positions of the *unpadded* image inside the
    padded (H+2p, W+2p) grid this layer's linear system was solved over.
    """
    if padding == 0:
        return np.ones(shape=in_shape, dtype=bool).squeeze()
    h, w = np.array(in_shape[-2:]) + 2 * padding
    toremain = np.ones(h * w * in_shape[1], dtype=bool)
    for c in range(in_shape[1]):
        for row in range(h):
            for col in range(w):
                if col < padding or w - col <= padding or row < padding or h - row <= padding:
                    i = c * h * w + row * w + col
                    assert toremain[i]
                    toremain[i] = False
    return toremain


def padding_constraints(in_shape, padding):
    """One equality row per zero-padded entry: enforces x[padded pos] = 0
    as extra rows of the linear system solved in `cnn_reconstruction`.
    """
    toremain = peeling(in_shape, padding)
    P = []
    for i in range(toremain.size):
        if not toremain[i]:
            P_row = np.zeros(toremain.size, dtype=np.float64)
            P_row[i] = 1
            P.append(P_row)
    return np.array(P)


def cnn_reconstruction(in_shape, k, g, out, kernel, stride, padding):
    """Recover a conv layer's (padded, flattened) input by stacking three
    sets of linear constraints and solving via least squares:
      - K @ x = g    (the observed weight gradient is bilinear in k, x)
      - W @ x = out  (the forward pass itself, out assumed already known
                       from inverting the next layer's activation)
      - P @ x = 0    (zero-padding positions must be zero)
    """
    coors, x_len, y_len = generate_coordinates(x_shape=in_shape, kernel=kernel, stride=stride, padding=padding)
    K = aggregate_g(k=k, x_len=x_len, coors=coors)
    W = circulant_w(x_len=x_len, kernel=kernel, coors=coors, y_len=y_len)
    P = padding_constraints(in_shape=in_shape, padding=padding)
    p = np.zeros(shape=P.shape[0], dtype=np.float64)
    if np.any(P):
        a = np.concatenate((K, W, P), axis=0)
        b = np.concatenate((g.reshape(-1), out, p), axis=0)
    else:
        a = np.concatenate((K, W), axis=0)
        b = np.concatenate((g.reshape(-1), out), axis=0)
    result = np.linalg.lstsq(a, b, rcond=None)
    print(f'[R-GAP] lstsq residual: {result[1]}, rank: {result[2]} -> {W.shape[-1]}, '
          f'max/min singular value: {result[3].max():.2e}/{result[3].min():.2e}')
    x = result[0]
    return x[peeling(in_shape=in_shape, padding=padding)], W


def fcn_reconstruction(k, gradient):
    """Recover a bias-free FC layer's input: since g[i,j] = k_i * x_j, every
    row i with k_i != 0 gives an independent estimate of x_j; average them.
    """
    x = [g / c for g, c in zip(gradient, k) if c != 0]
    x = np.mean(x, 0)
    return x


def r_gap(out, k, g, x_shape, weight, module):
    """Dispatch to the conv-specific linear system for one Conv2d layer."""
    if isinstance(module.layer, nn.Conv2d):
        padding = module.layer.padding[0]
        stride = module.layer.stride[0]
    else:
        padding = 0
        stride = 1

    x, weight = cnn_reconstruction(in_shape=x_shape, k=k, g=g, kernel=weight, out=out, stride=stride, padding=padding)
    return x, weight
