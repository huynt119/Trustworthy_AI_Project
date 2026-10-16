"""Measure the ResNet notebook's normalization mismatch before any long reconstruction.

The sample images are normalized with ImageNet mean/std. The reconstruction box
and the display/denormalization path use CIFAR-10 mean/std. This script does
not run GradientReconstructor.
"""
import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import inversefed
from inversefed.metrics import psnr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "part2_resnet_inverting_gradients", "data")
OUT = os.path.dirname(__file__)

im_mean = torch.tensor(inversefed.consts.imagenet_mean)
im_std = torch.tensor(inversefed.consts.imagenet_std)
cf_mean = torch.tensor(inversefed.consts.cifar10_mean)
cf_std = torch.tensor(inversefed.consts.cifar10_std)


def box(mean, std):
    return (-mean / std).tolist(), ((1 - mean) / std).tolist()


def denorm(normed, mean, std):
    return torch.clamp(normed * std[:, None, None] + mean[:, None, None], 0, 1)


path = os.path.join(DATA, "n02088364_beagle.JPEG")
raw = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])(Image.open(path).convert("RGB"))
normed = transforms.Normalize(im_mean.tolist(), im_std.tolist())(raw)

shown_cifar = denorm(normed, cf_mean, cf_std)
shown_imagenet = denorm(normed, im_mean, im_std)

# Notebook PSNR: MSE on normalized tensors, factor = 1/ds (per-channel, then averaged).
mse_norm = (normed - normed).pow(2).mean()  # perfect recon in normalized space
# A 1-gray-level pixel error in the normalized tensor, to show the formula gap.
err = normed + 0.05
mse_norm_err = (err - normed).pow(2).mean()
psnr_notebook = psnr(err.unsqueeze(0), normed.unsqueeze(0), factor=1 / cf_std[:, None, None])
psnr_notebook_im = psnr(err.unsqueeze(0), normed.unsqueeze(0), factor=1 / im_std[:, None, None])
pixel_cifar = denorm(err, cf_mean, cf_std)
pixel_im = denorm(err, im_mean, im_std)
mse_pix_cifar = (pixel_cifar - shown_cifar).pow(2).mean().item()
mse_pix_im = (pixel_im - shown_imagenet).pow(2).mean().item()
psnr_pix = lambda m: float("inf") if m == 0 else float(10 * torch.log10(torch.tensor(1.0 / m)))

report = {
    "input_normalize": "imagenet",
    "recon_dm_ds": "cifar10",
    "imagenet_mean": im_mean.tolist(),
    "imagenet_std": im_std.tolist(),
    "cifar10_mean": cf_mean.tolist(),
    "cifar10_std": cf_std.tolist(),
    "mean_abs_diff": (im_mean - cf_mean).abs().tolist(),
    "std_abs_diff": (im_std - cf_std).abs().tolist(),
    "box_cifar": {"low": box(cf_mean, cf_std)[0], "high": box(cf_mean, cf_std)[1]},
    "box_imagenet": {"low": box(im_mean, im_std)[0], "high": box(im_mean, im_std)[1]},
    "ground_truth_denorm_mse_cifar_vs_imagenet": (shown_cifar - shown_imagenet).pow(2).mean().item(),
    "ground_truth_denorm_max_abs": (shown_cifar - shown_imagenet).abs().max().item(),
    "fraction_of_pixels_clipped_by_cifar_denorm": (normed * cf_std[:, None, None] + cf_mean[:, None, None]).gt(1).logical_or(
        (normed * cf_std[:, None, None] + cf_mean[:, None, None]).lt(0)
    ).float().mean().item(),
    "synthetic_normalized_mse": mse_norm_err.item(),
    "notebook_psnr_with_cifar_factor": psnr_notebook,
    "notebook_psnr_with_imagenet_factor": psnr_notebook_im,
    "pixel_psnr_if_denorm_cifar": psnr_pix(mse_pix_cifar),
    "pixel_psnr_if_denorm_imagenet": psnr_pix(mse_pix_im),
    "formula_note": (
        "inversefed.metrics.psnr averages 10*log10((1/std_c)^2 / MSE) over channels. "
        "MSE is the normalized-tensor MSE, so this is not 10*log10(1/MSE_pixel)."
    ),
}
out = os.path.join(OUT, "resnet_preprocess.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
print("wrote", out)
