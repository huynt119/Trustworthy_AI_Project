"""Paired ablation for the LeNet Inverting Gradients PSNR gap.

The notebook's undefended run uses 24_000 x 2 and the true label. Each defense
cell uses 4_800 x 1, consumes RNG inside dp_sgd_defense, and passes labels=None.
At sigma=0 the inferred label equals the true label, but the budget does not.
This script keeps the part1/part2 LeNet, image 17, and original gradient fixed,
reseeds before every trial, and changes one factor at a time.
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, "/home/khuong/workspace/Trustworthy_AI_Project")
import inversefed

ROOT = "/home/khuong/workspace/Trustworthy_AI_Project"


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1), act(),
        )
        self.fc = nn.Sequential(nn.Linear(768, 10))

    def forward(self, x):
        out = self.body(x)
        return self.fc(out.view(out.size(0), -1))


torch.manual_seed(50)
setup = inversefed.utils.system_startup()
dst = datasets.CIFAR10(os.path.join(ROOT, "part1_dlg_lenet/data"))
tp = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
net = LeNet().to(**setup)
net.apply(weights_init)
gt = tp(dst[17][0]).to(**setup).view(1, 3, 32, 32)
label = torch.as_tensor((dst[17][1],), device=setup["device"])
net.zero_grad()
loss = F.cross_entropy(net(gt), label)
grad = [g.detach().clone() for g in torch.autograd.grad(loss, net.parameters())]
dm = torch.zeros(3, 1, 1, **setup)
ds = torch.ones(3, 1, 1, **setup)
base = dict(signed=True, boxed=True, cost_fn="sim", indices="top10", weights="equal",
            lr=0.1, optim="adam", total_variation=1e-6, init="randn", lr_decay=True,
            scoring_choice="loss")

rows = []
for name, iters, restarts, filt in [
    ("4800_median", 4_800, 1, "median"),
    ("24000_median", 24_000, 1, "median"),
    ("4800_none", 4_800, 1, "none"),
    ("24000_none", 24_000, 1, "none"),
]:
    torch.manual_seed(50)
    cfg = dict(base, restarts=restarts, max_iterations=iters, filter=filt)
    rec = inversefed.GradientReconstructor(net, (dm, ds), cfg, num_images=1)
    out, stats = rec.reconstruct(grad, label, img_shape=(3, 32, 32))
    mse = (out.detach() - gt).pow(2).mean().item()
    psnr = inversefed.metrics.psnr(out, gt, factor=1.0)
    row = {"name": name, "iters": iters, "restarts": restarts, "filter": filt,
           "rec_loss": stats["opt"], "mse": mse, "psnr": psnr}
    rows.append(row)
    print(json.dumps(row), flush=True)

dest = os.path.join(ROOT, "final-report/verification_gpu/ig_lenet_ablation.json")
with open(dest, "w") as f:
    json.dump(rows, f, indent=2)
print("wrote", dest)
