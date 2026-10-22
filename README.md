# PRIVET: PRoximIty leakage detection Via Extreme value Theory

PRIVET detects proximity-based data leakage (memorization, overfitting or data leakage occuring in underfitting regime) by auditing synthetic data sample by sample. It fits an extreme value law to the nearest-neighbour (1-NN) distances within the training set, then asks, for each synthetic sample, whether a generated sample is too close from a train sample.

## Install

For GPU nearest-neighbour search, first install the PyTorch build matching your CUDA version. Then:

```bash
pip install git+https://github.com/antoineszatkownik/PRIVET.git
```

## Usage

PRIVET has two stages:

1. **Fit.** 1-NN distances train→train ($`d^*_{TrTr}`$), EVD fit (Weibull or Gumbel) on a window of their lower tail. \
Plot the CDF and the fit to check the window.
2. **Score.** 1-NN distances synthetic→train ($`d^*_{STr}`$) and synthetic→test ($`d^*_{STe}`$), and scores for each synthetic sample. \
The CDF plot can now show $`d^*_{STr}`$ and $`d^*_{STe}`$.

**Remark:** Having a ```test``` set is not mandatory!

### Python

*Hyperparameter*
- Partition of the Train-Train 1-NN distances on which the EVT fit is applied

```python
from privet import PRIVET, plot_cdfs, plot_score_distributions

device = "cuda" #or "cpu"

# Stage 1
privet = PRIVET(train, device, partition_fit_start=0.01,    partition_fit_end=0.2,
                distance="hamming",          # or "standard_euclidean"
                fit_method="ols")            # or "mle"
plot_cdfs(privet)
#privet.fit(0.02, 0.3)                        # another window, distances not recomputed

# Stage 2: one row per synthetic sample, sorted by rank r
scores = privet.compute_scores(synthetic, test, score_fraction=0.5, threshold=None)
scores.to_csv("scores.csv", index=False)
plot_cdfs(privet, scores)

# Stage 3
privet.summary(scores)                       # N_overfit, N_pleaks, Δ score quantiles
plot_score_distributions(scores)
```

`train`, `test` and `synthetic` are numpy arrays or torch tensors with shape $`(N,d)`$ where $`N`$ is dataset size and $`d`$ is the number of features. Their sizes can differ! The test set is optional (no $`\Delta`$ scores and no $`N_{pleaks}`$ without it).

**Remark:** PRIVET is a statement about the tail. Depending on the data, the closer to the bulk the sample is, the less PRIVET becomes reliable as it involves extrapolating the extreme value law outside of the lower tail. At `score_fraction=0.5`, we assign a data leakage risk score for the first half of the data.

### Command line

```bash
privet fit --train train.npy --distance hamming --fit-window 0.01 0.2 --device cuda --out runs/fit
privet run --train train.npy --test test.npy --synthetic synth.npy --distance hamming --fit-window 0.01 0.2 --device cuda --out runs/audit
```

`fit` writes `cdfs.png`. `run` writes `scores.csv`, `summary.json`, `cdfs.png`, `score_distributions.png`. Data files: `.npy`, `.csv` or `.pt`.

## Outputs

**TL;DR:** \
The quantities to rely on are $`N_{overfit}`$ and $`N_{pleaks}`$ (available in the output summary), and the per-sample $`p_r^{train}`$ and $`\Delta p_r`$ (available in the output table). The lower $`p_r^{train}`$ and $`\Delta p_r`$ are, the higher the risk of leakage.

For a reference set (```train``` or ```test```) of size $`N`$, $`u_1 \leq \cdots \leq u_M`$ are the sorted 1-NN distances of the $`M`$ synthetic samples to it, $`\hat G(u) = 1 − \exp(−N A u^\alpha)`$ (Weibull) or $`1 − \exp(−N A e^{Bu})`$ (Gumbel) the fitted law at size $`N`$, and $`r[u] = \lfloor M Ĝ(u)\rfloor`$ the rank expected for distance $`u`$ (paper, Sec. 2.3 and Appendix B).

| Column | Definition |
|---|---|
| `rank`, `syn_idx` | rank $`r`$ upon sorting, and the row of the sample in `synthetic` |
| `d_train`, `nn_train_idx` | $`u_r^{train}`$ and the row of the nearest training sample |
| `d_test`, `nn_test_idx`, `rank_test` | the same w.r.t. test; $`r'`$ |
| `fit_region` | `d_train` lands below, inside or above the fit window |
| `G_train`, `G_test` | $`\hat G`$ at `d_train` (train size) and at `d_test` (test size) |
| `n_overfit` | $`r - r[u_r^{train}]`$; number of overfit samples up to that rank |
| `n_pleaks` | $`r[u_r^{test}] - r[u_r^{train}]`$; number of leaked samples up to that rank |
| `log10_p_train`, `log10_p_test`, `delta_p` | cf. eq (7)


## Threshold

`compute_scores(..., threshold=None)` adds a `flagged` column when threshold is not `None`: scored samples with $`\Delta p_r \leq`$ threshold, or $`\log_{10} p_r^{train} \leq`$ threshold without a test set.

Two ways to tune the threshold:
- Using real data, perform a train, test, pseudo-synthetic split. This allows to create a no-leakage scenario. Calibrate the threshold for the pseudo-synthetic based on a desired false positive rate (e.g. 1% or 5%).
- If data is abundant, use an additional independent validation split to find the optimal threshold based on the pareto front of ROC-PR curves.

## Fit window

The quantile-based window is the fraction of sorted train→train distances used for the fit. Windows used in the paper: haplotypes [1%, 20%]; CIFAR-10  [0.1%, 2%]; ImageNet  [0.1%, 10%]; UCI Adult / Bank [0.1%, 2.8%] / [0.2%, 10%].

