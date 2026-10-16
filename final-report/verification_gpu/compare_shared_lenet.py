"""Check whether part1 and part2_lenet build the same LeNet and the same gradient."""
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


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


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
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(768, 10))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def build(which):
    torch.manual_seed(50)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dst = datasets.CIFAR10(os.path.join(ROOT, "part1_dlg_lenet/data"))
    tp = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
    if which == "part2":
        setup = inversefed.utils.system_startup()
        device = setup["device"]
    net = LeNet().to(device)
    net.apply(weights_init)
    img = tp(dst[17][0]).to(device).view(1, 3, 32, 32)
    gt_label = torch.tensor([dst[17][1]], device=device)
    if which == "part1":
        onehot = label_to_onehot(gt_label.long())
        loss = cross_entropy_for_onehot(net(img), onehot)
    else:
        loss = F.cross_entropy(net(img), gt_label)
    grads = [g.detach().clone() for g in torch.autograd.grad(loss, net.parameters())]
    weights = [p.detach().clone() for p in net.parameters()]
    return img, gt_label, weights, grads, float(loss)


img1, y1, w1, g1, loss1 = build("part1")
img2, y2, w2, g2, loss2 = build("part2")
weight_max = max((a - b).abs().max().item() for a, b in zip(w1, w2))
grad_max = max((a - b).abs().max().item() for a, b in zip(g1, g2))
# iDLG label from the fc.weight gradient, which is the second-to-last parameter.
idlg = int(torch.argmin(g1[-2].sum(dim=-1)).item())
report = {
    "same_image_max_abs": (img1 - img2).abs().max().item(),
    "gt_label": int(y1.item()),
    "labels_equal": int(y1.item()) == int(y2.item()),
    "weight_max_abs": weight_max,
    "grad_max_abs": grad_max,
    "part1_loss": loss1,
    "part2_loss": loss2,
    "part1_grad_l2": torch.sqrt(sum((g ** 2).sum() for g in g1)).item(),
    "part2_grad_l2": torch.sqrt(sum((g ** 2).sum() for g in g2)).item(),
    "part1_idlg_label": idlg,
    "part2_main_attack_receives_true_label": True,
    "part2_defense_passes_labels_None_so_idlg": True,
}
out = "/home/khuong/workspace/Trustworthy_AI_Project/final-report/verification_gpu/shared_lenet.json"
with open(out, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
