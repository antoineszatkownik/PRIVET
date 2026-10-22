#-------------------------------------------------------------------------------
# PRIVET
#
# Stage 1  PRIVET(train, ...)            1-NN distances train -> train, EVD fit
# Stage 2  privet.compute_scores(S, Te)  1-NN distances synthetic -> train / test,
#                                        one row of scores per synthetic sample
# Stage 3  privet.summary(scores), plot_utils
#
# Notation (paper Sec. 2.3 and Appendix B). M synthetic samples; for a reference
# set (train or test) of size n_ref, u_1 <= ... <= u_M are the sorted 1-NN
# distances of the synthetic samples to it, \hat G the fitted law at size n_ref,
# r[u] = floor(M \hat G(u)) the rank expected for distance u, and
#     pi_r          = P(Binom(M, \hat G(u_r)) >= r)
#     n_excess(r)   = r - r[u_r],                n_excess(0) = 0
#     rho_r         = r - n_excess(r-1) = 1 + r[u_{r-1}]
#     p_r           = P(Binom(M, \hat G(u_r)) > rho_r) = pi_{rho_r + 1}
#                     (as in the code behind the paper's figures; the text writes pi_{rho_r})
#     n_overfit(r)  = r - r[u_r^train]
#     n_pleaks(r)   = r[u_r^test] - r[u_r^train]
#     Delta p       = log10 p_r^train - log10 p_r'^test     (r' = rank of the same sample w.r.t. test)
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch

from .nn_utils import gpu_nearest_neighbors
from .stats_utils import fit_evd, log10_sf


class PRIVET:
    """
    Parameters
    ----------
    train : np.ndarray or torch.Tensor, shape (N_train, D)
        Training data, one row per sample (images etc. are flattened).
    device : "cuda", "cuda:0", "cpu", ...
    partition_fit_start, partition_fit_end : float
        Fit window, as fractions of the sorted train->train 1-NN distances.
        Choose it from plot_utils.plot_cdfs.
    distance : "standard_euclidean", "feature_normalized_euclidean" or "hamming"
    fit_method : "ols" (least squares on the linearised eCDF) or "mle"
        (censored maximum likelihood).
    loo_correction : each train->train 1-NN distance is a minimum over the N_train - 1
        other training points, so the fit estimates (N_train - 1) A, not N_train A.
        True divides by N_train - 1; False (default) by N_train, as in the paper.
        The difference in ln A is ln(N/(N-1)) ~ 1/N_train.
    seed : seed of the random order given to synthetic samples at equal distance.
    """

    def __init__(self, train, device="cuda", partition_fit_start=0.01, partition_fit_end=0.2,
                 distance="standard_euclidean", fit_method="ols", loo_correction=False, seed=0):
        self.device = device
        self.distance = distance
        self.loo_correction = loo_correction
        self.seed = seed

        self.train = self._as_tensor(train)
        self.N_train = self.train.shape[0]

        # ——— Stage 1: 1-NN distances among training points, EVD fit ———
        self.d_tr_tr = self._self_distances(self.train)          # d*_TrTr, sorted
        self.fit(partition_fit_start, partition_fit_end, fit_method)

    def fit(self, partition_fit_start, partition_fit_end, fit_method="ols"):
        """(Re)fit the EVD on the stored train->train distances, e.g. on a new window."""
        self.partition_fit = (partition_fit_start, partition_fit_end)
        self.fit_method = fit_method
        self.law = fit_evd(self.d_tr_tr, partition_fit_start, partition_fit_end, fit_method,
                           loo_correction=self.loo_correction)
        return self

    # ------------------------------------------------------------------
    def compute_scores(self, synthetic, test=None, score_fraction=0.5, threshold=-1.0,
                       upper_tail="extrapolate", test_reference="train_law"):
        """
        Stage 2. One row per synthetic sample, sorted by rank r of its 1-NN
        distance to train.

        score_fraction : only ranks r <= ceil(score_fraction * M) get scores (NaN
            elsewhere): PRIVET is a statement about the tail.
        threshold : flagged = scored and delta_p <= threshold (log10_p_train without
            a test set). None: no flagged column.
        upper_tail : above the fit window, "extrapolate" the fit, 
            or read \hat G there off the "empirical" CDF of the reference set's own 1-NN distances
        test_reference : how distances to test are read.
            "train_law" (default): the train law at size N_test, Lambda proportional to
                the size. Exact in the deep tail; biased further up when the density of
                the data is heterogeneous and N_test differs from N_train.
            "test_law": a law fitted on test->test 1-NN distances (same window, method
                and family). Equivalent to mapping distances to test with
                Q = \hat G_TrTr^{-1} o \hat G_TeTe and reading them with the train law
        """
        synthetic = self._as_tensor(synthetic)
        M = synthetic.shape[0]
        rng = np.random.default_rng(self.seed)

        d_s_tr, idx_s_tr = self._nearest(synthetic, self.train)
        tr = self._rank_statistics(d_s_tr, self.law, self.N_train, upper_tail, rng)
        o = tr["order"]                                   # o[r-1] = sample at train rank r
        r = np.arange(1, M + 1)

        scores = pd.DataFrame({
            "rank": r,
            "syn_idx": o,
            "d_train": tr["u"],
            "nn_train_idx": idx_s_tr[o],
            "fit_region": np.where(tr["u"] < self.law.window[0], "below",
                                   np.where(tr["u"] > self.law.window[1], "above", "inside")),
            "G_train": tr["G"],
            "log10_p_train": tr["log10_p"],
            "n_overfit": r - tr["r_hat"],
        })

        if test is not None:
            test = self._as_tensor(test)
            N_test = test.shape[0]
            if test_reference == "train_law":
                law_test = self.law
            elif test_reference == "test_law":
                law_test = self._test_law(test)
            else:
                raise ValueError("test_reference must be 'train_law' or 'test_law'")
            self.law_test = law_test

            d_s_te, idx_s_te = self._nearest(synthetic, test)
            te = self._rank_statistics(d_s_te, law_test, N_test, upper_tail, rng)
            #evaluating the train law at the test size is equivalent to rescaling the distances
            #\hat G_{N_te}(u) = 1 − exp(−N_te*A*u^alpha) = 1 − exp(−N_tr*A*(kappa*u)^alpha)
            #                 = \hat G_{N_tr}(kappa*u), with kappa = (N_te/N_tr)^{1/alpha} = lambda^{−1/alpha},
            #likewise for Gumbel with the shift u − ln(lambda)/B.

            def at_train_rank(x_by_test_rank):
                """test-side quantity of each sample, listed in train-rank order"""
                x_by_sample = np.empty_like(x_by_test_rank)
                x_by_sample[te["order"]] = x_by_test_rank
                return x_by_sample[o]

            scores["d_test"] = d_s_te[o]
            scores["nn_test_idx"] = idx_s_te[o]
            scores["rank_test"] = at_train_rank(r)                       # r'
            scores["G_test"] = at_train_rank(te["G"])
            scores["log10_p_test"] = at_train_rank(te["log10_p"])
            scores["delta_p"] = scores["log10_p_train"] - scores["log10_p_test"]
            scores["n_pleaks"] = np.maximum(0,te["r_hat"] - tr["r_hat"])   # both read at the same rank r
            scores.attrs["test_law"], scores.attrs["N_test"] = law_test, N_test   # for plot_cdfs

        # no score beyond the tail fraction
        not_scored = r > np.ceil(score_fraction * M)
        score_cols = [c for c in scores.columns
                      if c.startswith(("G_", "log10_", "delta_", "n_"))]
        scores[score_cols] = scores[score_cols].astype(float)
        scores.loc[not_scored, score_cols] = np.nan

        if threshold is not None:
            score = "delta_p" if test is not None else "log10_p_train"
            scores["flagged"] = scores[score] <= threshold      # False where not scored
            scores.attrs["threshold"] = f"{score} <= {threshold}"
        return scores

    def _rank_statistics(self, d, law, n_ref, upper_tail, rng):
        """Order statistics of the synthetic 1-NN distances to one reference set,
        all arrays indexed by rank r = 1..M."""
        M = len(d)
        order = np.lexsort((rng.random(M), d))   # sort by distance. when data is discrete, samples can be at same distance, sorting breaks down then
        u = d[order].astype(float)               # u_1 <= ... <= u_M
        G = law.cdf(u, n_ref, upper_tail)        # \hat G(u_r)
        r = np.arange(1, M + 1)
        r_hat = np.floor(M * G).astype(int)      # r[u_r]
        rho = np.ones(M, dtype=int)              # rho_r = r - n_excess(r-1) = 1 + r[u_{r-1}]
        rho[1:] = 1 + r_hat[:-1]                 # rho_1 = 1 since n_excess(0) = 0
        rho = np.minimum(rho, M - 1)
        return {
            "order": order, "u": u, "G": G, "r_hat": r_hat,
            "log10_p": log10_sf(rho, M, G),           # p_r = P(X > rho_r) = pi_{rho_r + 1}
        }

    # ------------------------------------------------------------------
    def summary(self, scores):
        """Stage 3: global results on the scored samples. N_overfit and N_pleaks are
        the maxima over the scored ranks of n_overfit(r) and min(n_pleaks(r), r):
        among the first r samples there cannot be more than r leaks."""
        law = self.law
        s = scores.dropna(subset=["log10_p_train"])           # scored samples
        out = {
            "N_train": self.N_train, "N_synthetic": len(scores),
            "fit": f"{law.family} ({law.fit_method}), ln A = {law.log_A:.4g}, "
                   f"{'alpha' if law.family == 'Weibull' else 'B'} = {law.param2:.4g}",
            "fit_window": f"{self.partition_fit} -> d in [{law.window[0]:.4g}, "
                          f"{law.window[1]:.4g}]",
            "n_scored": len(s),
            "scored_vs_fit_window": s["fit_region"].value_counts().to_dict(),
            "N_overfit": int(s["n_overfit"].max()),
            "rank_N_overfit": int(s.loc[s["n_overfit"].idxmax(), "rank"]),
        }
        score = "log10_p_train"
        if "n_pleaks" in s:
            n_pleaks = np.minimum(s["n_pleaks"], s["rank"])
            out["N_pleaks"] = int(n_pleaks.max())
            out["rank_N_pleaks"] = int(s.loc[n_pleaks.idxmax(), "rank"])
            score = "delta_p"
        q = s[score].quantile([0.0, 0.01, 0.05, 0.5]).round(3)
        out[f"{score} min / 1% / 5% / median"] = list(q)
        if "flagged" in scores:
            out[f"flagged ({scores.attrs.get('threshold', 'threshold')})"] = \
                int(scores["flagged"].sum())
        return out

    # ------------------------------------------------------------------
    def _as_tensor(self, x):
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
        return x.reshape(x.shape[0], -1)

    def _nearest(self, src, dst):
        d, idx = gpu_nearest_neighbors(src, dst, k=1, distance=self.distance,
                                       device=self.device)
        return d.numpy().reshape(-1), idx.numpy().reshape(-1)

    def _test_law(self, test):
        """EVD fitted on test->test distances; cached, since the same test set is
        usually scored against many synthetic sets."""
        key = (tuple(test.shape), float(test.double().sum()), float((test.double() ** 2).sum()),
               self.partition_fit, self.fit_method, self.law.family)
        if getattr(self, "_test_law_cache", (None,))[0] != key:
            law = fit_evd(self._self_distances(test), *self.partition_fit, self.fit_method,
                          family=self.law.family, loo_correction=self.loo_correction)
            self._test_law_cache = (key, law)
        return self._test_law_cache[1]

    def _self_distances(self, X):
        """sorted leave-one-out 1-NN distances within X"""
        return np.sort(self._nearest(X, None)[0])
