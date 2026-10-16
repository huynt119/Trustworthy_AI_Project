# GPU verification of the original notebooks

Date: 2026-10-16. Python 3.12.12. PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130.
Device: NVIDIA GeForce RTX 4060 Laptop GPU. The team notebooks were executed on
an RTX 5060 Ti with PyTorch 2.10.0+cu128. Each notebook was executed from a fresh
interpreter. The Windows `E:\...` export path was redirected into this folder.
Attack budgets, seeds, models, and labels were not changed.

## Configurations read before execution

| Notebook | Data | Seed | Model | Label the attacker receives | Optimizer | Steps |
|---|---|---|---|---|---|---|
| `part1.ipynb` | CIFAR-10 index 17, class 3 (cat), pixels in [0, 1] | 50 | LeNet, sigmoid, uniform(-0.5, 0.5) | DLG optimizes a dummy label. iDLG reads it from the sign of the last weight gradient | LBFGS | 300; defense 50 |
| `part2_lenet.ipynb` | same image and LeNet | 50 | same | Main call receives the true label. Defense passes `None`, so iDLG infers it | Adam, lr 0.1, signed, median | 24,000 x 2; defense 4,800 x 1 |
| `part2.ipynb` | ImageNet sample beagle, class 162, ImageNet mean/std | model default init | untrained ResNet-18 | same split as LeNet IG | same Adam setup | 24,000 x 4; defense 4,800 x 1 |
| `part1_batch2.ipynb` | CIFAR-10 indices starting at 6, classes 2 and 7 | 50 | same LeNet | DLG optimizes both labels. iDLG takes the two most negative class rows | LBFGS | 300; defense 50 |
| `part2_batch2.ipynb` | beagle and great grey owl | default init | untrained ResNet-18 | true labels on the main call; iDLG on defense | Adam | 24,000 x 2; defense 4,800 x 1 |
| `part3_batch2.ipynb` | same two CIFAR images; FCN3 sees them resized to 28 | 50 | LeNetRGAP and FCN3 | attack label is the model's own flipped logit, not the CIFAR class | closed-form R-GAP | one solve |

`part1` and `part2_lenet`, reseeded separately, produce identical weights
(max abs 0) and the same gradient up to 9e-8. Global L2 is 29.4451 on this GPU
and 29.4434 in the team notebook. One-hot cross-entropy and `F.cross_entropy`
match on this single image.

## ResNet preprocessing, measured before the long run

Inputs are normalized with ImageNet statistics. The reconstruction box and
`plot`/`to_img` denormalize with CIFAR-10 statistics. On the beagle, 8.0% of
normalized pixels fall outside the CIFAR box. Projecting them back changes the
true pixels by MSE 1.4e-4 (about 39 dB), so the wrong box is not what limits a
17 dB reconstruction. Denormalizing the true image with CIFAR statistics instead
of ImageNet changes the displayed ground truth by MSE 0.00163, max abs 0.118,
and clips 8% of pixels.

`test_mse` is the normalized-tensor MSE. `inversefed.metrics.psnr(..., factor=1/ds)`
averages `10 log10((1/std_c)^2 / MSE)` over channels. That is about 12 dB above
`10 log10(1/MSE_normalized)`. Peak-1 PSNR after the notebook's own denormalization
lands next to the factor-based number, and that denormalized MSE is the 0.0185
published in the report.

This was not patched in the notebooks. Switching `dm`/`ds` to ImageNet statistics
moves the box, so the finished 24,000-iteration runs would no longer describe the
edited code. The pixel floor above is why that second reconstruction was not started.

## Results against the stored team outputs

### Part 1

Labels match: dummy init 7, DLG recovers 3, iDLG recovers 3, every defense label is correct.

| | Team MSE | This MSE |
|---|---:|---:|
| DLG | 0.00140 | 0.00408 |
| iDLG | 0.00160 | 0.00385 |

iDLG's gradient-match loss starts at 12.90 here and 12.14 in the team log, and
finishes at 3.0e-4 versus 9.8e-5. LBFGS on PyTorch 2.14 does not retrace the
2.10 path, so the 28.55 / 27.95 dB baseline is not reproduced as a number.
The defense pattern is: half-norm clipping still throws the 50-step iDLG iterate
far outside [0, 1] (PSNR -90 dB at sigma 0; the team run was -105 dB), and with
C equal to the gradient norm the PSNR falls 14.8, 12.7, -7.6, -19.5 dB across
sigma = 0, 1e-4, 1e-3, 1e-2 (team: 16.1, 14.0, -8.1, -18.8).

### Part 2 LeNet

Main run: MSE 0.0023, PSNR 26.38 dB, 462 s. Team: MSE 0.0022, PSNR 26.48 dB.
The main call was given the true label. Defense infers the label and, at sigma 0,
infers it correctly.

Defense PSNR at sigma 0 is 28.21 dB (half clip) and 28.06 dB (no clip), above the
24,000-iteration baseline, as in the team table (28.60 and 27.94). A paired rerun
with the same weights, the same gradient, the true label, and a fresh seed before
each trial separates the budget from the filter:

| Iterations | Median every 500 | No filter |
|---:|---:|---:|
| 4,800 | 27.85 dB | 16.33 dB |
| 24,000 | 26.36 dB | 22.06 dB |

The higher defense PSNR is the 4,800-step median budget, not a benefit of clipping
or noise. Median filtering improves pixel error relative to the unfiltered run.
Extending that filtered run from 4,800 to 24,000 lowers PSNR.

### Part 2 ResNet-18

Mean per-tensor gradient norm 3.300884e+00 versus the team's 3.300741e+00.
Iteration 500 reconstruction loss 0.0270 versus 0.0271.

| Quantity | This run | Report |
|---|---:|---:|
| Normalized MSE (`test_mse`) | 0.3013 |  |
| Notebook PSNR (`factor=1/ds`) | 17.23 dB | 17.33 dB |
| MSE after CIFAR denorm | 0.01840 | 0.0185 |
| Peak-1 PSNR after that denorm | 17.35 dB |  |

The published pair is the denormalized pair. It is not `test_mse` with
`PSNR = 10 log10(1/MSE)`. Defense PSNR matches the team table within 0.06 dB
at every grid point (12.69/12.14/9.75/9.35 and 12.69/12.16/9.65/9.46 versus
12.66/12.18/9.71/9.40 and 12.72/12.17/9.71/9.43).

### Batch 2 and R-GAP

DLG/iDLG labels match exactly, including dummy init [6, 3] and recovery [2, 7].
Mean MSE is 0.0273 / 0.0253 versus the team's 0.0235 / 0.0216. Half-norm clipping
no longer diverges to 1e10; both runs sit near 0 dB, and the unclipped curve falls
from about 15 dB to about -18 dB.

ResNet batch 2: normalized MSE 0.375, notebook PSNR 16.29 dB, denormalized MSE
0.0233, peak-1 PSNR 16.33 dB. The report's 16.35 dB is the denormalized figure.
Images are beagle and great grey owl.

R-GAP batch 2 reproduces the stored cosine similarities to four decimals.
LeNet final-image cosine to the average is 0.0608 (versus image A 0.0617, B 0.0550).
FCN3's final cosine to the average is 1.0000 (A 0.9591, B 0.9665). Attack labels
are [0, 0] for the CNN and [1, 1] for FCN3; CIFAR classes [2, 7] are unused.
The last CNN least-squares residual is 58.29, matching the stored 58.289.
CNN wall time was 196 s here and 59.5 s in the team notebook.

### Part 3 on GPU

`part3/` replaces the earlier CPU sweep. Device is CUDA. Baseline float64 MSE
before the float32 cast is 2.84e-22 (PSNR 215.47 dB); after the cast, MSE is 0
and PSNR is infinite. The float32 attack collapses to MSE 91.00 and PSNR -19.59 dB,
matching the stored -19.60 dB. The 16-point defense grid matches the report,
including 12.82 dB with half-norm clipping and -19.53 dB without clipping at
sigma 1e-6. No-clip PSNR across the first six noise levels is inf, 73.22,
51.57, 31.02, -19.53, -18.53 dB. Least-squares solves still run in NumPy, so
one configuration took 6876 s; the others took about 110--170 s. Profiled
baseline: 127 s total, 12 s scalar solve, 112 s least squares.
