# Findings audit and separate CPU verification

Date: 2026-10-16. Python 3.12.12; PyTorch 2.14.1+cu130;
torchvision 0.29.1+cu130; NumPy 2.5.2; device CPU; torch threads 14.
The CUDA-enabled package build does not mean a GPU was used.

## Scope and provenance

Executed Part 3 batch-1 imports, CIFAR-10 image 17, model initialization,
float64 gradient, baseline attack, intermediate plots, float32 precision test,
all 16 original defense configurations, result plots and separate exports.
An extra baseline profiling cell times the scalar and NumPy least-squares solves.
Original attack budgets and weights were not changed. Data download time is excluded.
The profiling run consumes no random numbers; the original float32-test model
initialization still precedes the defense sweep. Historical GPU/backend noise
realizations need not match CPU draws, even with the same seed.

Other notebooks and controlled IG median/normalization ablations have NOT been rerun
in this verification. Historical report figures/tables are not replaced by this run.
No LaTeX compiler was found, so edited sources have NOT been compiled to PDF.

## Findings

1. **Runtime:** historical R-GAP output was 57.1 s, not 57.3 s.
   This CPU baseline measured 75.914 s; a separate profile measured 84.125 s:
   scalar Adam 2.993 s, four `lstsq` calls 78.488 s, other work 2.644 s.
   Thus neither scalar dominance nor instantaneous linear solves is supported here.
   Stored DLG/iDLG cell elapsed estimates were about 156/148 s, not a few seconds;
   they are not identically instrumented attack timings. IG LeNet printed 532.355 s.
   Do not infer a universal ranking or compare CPU and GPU wall times directly.

2. **IG PSNR:** stored baseline 26.48 dB and lower-budget defense values
   27.94/28.60 dB are real. Median filtering every 500 iterations is implemented,
   but smoothing as the cause is unproven. Budgets, restart initialization and
   label handling differ. A paired-seed ablation is required.

3. **Clipped R-GAP anomaly:** the historical grid has 12.82 dB clipped versus
   -19.53 dB nominally unclipped at sigma=1e-6. The new CPU run reverses that:
   -14.612 dB clipped versus 14.656 dB nominally unclipped. This is not a
   reproduced clipping advantage. Noise amplitude, realization, recursive
   activation branches, and scalar solver residuals must be separated before
   claiming conditioning improvement. `C` is rounded to six decimals in the
   original code; the nominal no-clip setting is not mathematically guaranteed
   to equal the exact norm on every run.

4. **Normalization:** both ResNet notebooks use ImageNet input statistics but
   CIFAR-10 reconstruction/display statistics. This affects box constraints and
   denormalization; claiming a negligible effect requires measurement. Their MSE
   is computed on normalized tensors, while PSNR uses a scaling factor, so the
   report's blanket pixel-space metric definition also needs reconciliation.
   The mismatch was not silently fixed, to avoid invalidating historical results.

5. **Presentation:** added the five report authors to the slide title page.
   Corrected ResNet-50 to ResNet-18 in both Part 1 introductions and clarified
   that ResNet uses ImageNet samples, not the same CIFAR-10 dataset.

## Additional reproducibility issues found by execution

- Baseline MSE after float32 casting is zero, giving infinite PSNR. Before the
  cast, float64 MSE is 2.71595e-22 and PSNR is 215.661 dB. Describe this as
  numerical agreement, not mathematically exact recovery.
- The historical float32 collapse does not reproduce: this CPU test reaches
   66.57 dB, with an identifiable image. Old notebook prose/precision plot title
  still describe the historical collapse and must not be read as this run's result.
- The original plot labeled infinite PSNR as `nan`; the defense plot now handles
  positive infinity explicitly.

## Selected CPU sweep results

| sigma | half-norm clipping PSNR (dB) | nominal no clipping PSNR (dB) |
|---|---:|---:|
| 0 | 14.183 | inf (after float32 cast) |
| 1e-9 | 14.183 | 70.450 |
| 1e-8 | 14.183 | 53.897 |
| 1e-7 | -13.613 | 31.960 |
| 1e-6 | -14.612 | 14.656 |
| 1e-5 | -21.564 | -22.703 |
| 1e-4 | -23.015 | -29.754 |
| 1e-3 | -36.414 | -45.466 |

Sweep runtime was about 21.7 minutes. Full scalar timings and full-precision
baseline metrics are in rgap_baseline.json; precision summary is in
rgap_precision.json; all sweep measurements are in rgap_defense.csv.
part3.npz includes unclamped reconstructions, weights, gradients, timings and
the post-sweep CPU RNG state. Object-array weights/gradients require trusted
`allow_pickle=True` to load. Images are separate CPU figures, not replacements
for the report's historical images.