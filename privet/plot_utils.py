import matplotlib.pyplot as plt
import numpy as np

TRTR, FIT, STR, STE, WINDOW = "olive", "#009E73", "#C2185B", "#0288A6", "#E69F00"


def _ecdf(d):
    d = np.sort(np.asarray(d, dtype=float))
    return d, np.arange(1, len(d) + 1) / len(d)


def plot_cdfs(privet, scores=None, ax=None):
    """
    eCDFs of 1-NN distances with the EVD fit and the fit window.
    Stage 1: d*_TrTr only. Stage 2: pass the output of compute_scores to add
    d*_STr and d*_STe.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    law = privet.law
    ax.axvspan(*law.window, color=WINDOW, alpha=0.2, lw=0,
               label=f"fit window {privet.partition_fit}")

    curves = [(privet.d_tr_tr, TRTR, r"$d^*_{TrTr}$")]
    if scores is not None:
        curves.append((scores["d_train"], STR, r"$d^*_{STr}$"))
        if "d_test" in scores:
            curves.append((scores["d_test"], STE, r"$d^*_{STe}$"))
    for d, color, label in curves:
        x, y = _ecdf(d)
        ax.plot(x, y, ".", ms=2, color=color, label=label)

    d_all = np.concatenate([np.asarray(c[0], dtype=float) for c in curves])
    d_all = d_all[d_all > 0]
    u = np.linspace(d_all.min(), d_all.max(), 500)
    ax.plot(u, law.cdf(u, privet.N_train), "--", color=FIT,
            label=f"{law.family} fit ({law.fit_method})")
    if scores is not None and "test_law" in scores.attrs:
        law_te, n_te = scores.attrs["test_law"], scores.attrs["N_test"]
        if law_te is not law or n_te != privet.N_train:   # reference law of d*_STe
            ax.plot(u, law_te.cdf(u, n_te), ":", color=STE, label=f"test reference (N = {n_te})")

    if d_all.max() / d_all.min() > 10:      # log x-axis only if it spans a decade
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())   # no overlapping minor labels
    ax.set_yscale("log")
    ax.set_ylim(0.5 / privet.N_train, 1.5)
    ax.set_xlabel("1-NN distance")
    ax.set_ylabel("eCDF")
    ax.legend(loc="upper left", frameon=False, markerscale=4, fontsize=9)
    return ax


def plot_score_distributions(scores, threshold=None):
    """Histograms of the scored samples' log10 p_r^train and, with a test set, Delta p_r."""
    cols = ["log10_p_train"] + (["delta_p"] if "delta_p" in scores else [])
    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 3.5), squeeze=False)
    for ax, col in zip(axes[0], cols):
        v = scores[col].dropna().to_numpy()
        finite = v[np.isfinite(v)]
        ax.hist(finite, bins=60, color=STE)
        if len(finite) < len(v):
            ax.text(0.02, 0.95, f"{len(v) - len(finite)} at -inf", transform=ax.transAxes,
                    va="top", fontsize=9)
        if threshold is not None and col == cols[-1]:
            ax.axvline(threshold, color="black", ls="--", lw=1, label=f"threshold {threshold}")
            ax.legend(frameon=False, fontsize=9)
        ax.set_yscale("log")      # the tail of the distribution is what matters
        ax.set_xlabel(col)
        ax.set_ylabel("synthetic samples")
    fig.tight_layout()
    return fig
