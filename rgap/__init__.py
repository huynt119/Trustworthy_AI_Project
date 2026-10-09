"""R-GAP (Recursive Gradient Attack on Privacy) — closed-form, non-optimization
gradient inversion, ported from https://github.com/JunyiZhu-AI/R-GAP:

    Junyi Zhu, Matthew B. Blaschko. "R-GAP: Recursive Gradient Attack on
    Privacy." ICLR 2021.

Used by ``part3_rgap_lenet/part3.ipynb`` as the "closed-form" counterpart to
the optimization-based DLG/iDLG (part1) and Inverting Gradients (part2)
attacks in this project.
"""
from .models import LeNetRGAP, FCN3, weights_init
from .recursive_attack import r_gap, peeling, fcn_reconstruction, inverse_udldu
from .utils import (logistic_loss, inverse_leakyrelu, derive_leakyrelu,
                     inverse_identity, derive_identity, inverse_sigmoid, derive_sigmoid)
