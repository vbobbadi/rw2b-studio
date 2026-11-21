"""Statistics: paired bootstrap on the same test samples, and aggregation across seeds.

Within one seed, notebook 05's bootstrap CIs are tight (942 test samples), but they ignore training
noise. Seed aggregation answers the question the progress report raised: is a ~2 mm gap real?
A difference counts as robust only if it has the same sign in every seed AND the CI of the
seed-averaged per-sample difference excludes zero.
"""

import numpy as np


def paired_bootstrap(a, b, n=5000, seed=0):
    """Mean of (a - b) with a 95% CI. Negative = a has lower error."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def seed_comparison(a_by_seed, b_by_seed, n=5000, seed=0):
    """a_by_seed, b_by_seed: lists (one per seed) of per-sample errors on the SAME test samples.
    Returns per-seed means, mean +- std across seeds, and a bootstrap CI that resamples test samples on
    the seed-averaged difference (two-level: also resamples seeds)."""
    A = np.stack([np.asarray(x, np.float64) for x in a_by_seed])     # (S, N)
    B = np.stack([np.asarray(x, np.float64) for x in b_by_seed])
    D = A - B
    per_seed = D.mean(1)
    rng = np.random.default_rng(seed)
    S, N = D.shape
    boots = np.empty(n)
    for i in range(n):
        s_idx = rng.integers(0, S, S)
        n_idx = rng.integers(0, N, N)
        boots[i] = D[s_idx][:, n_idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    same_sign = bool(np.all(per_seed < 0) or np.all(per_seed > 0))
    return {
        "diff_mean": float(per_seed.mean()), "diff_std_across_seeds": float(per_seed.std(ddof=1)) if S > 1 else 0.0,
        "per_seed": [round(float(x), 2) for x in per_seed], "ci_lo": float(lo), "ci_hi": float(hi),
        "same_sign_all_seeds": same_sign, "robust": same_sign and (hi < 0 or lo > 0),
    }


def mean_std_table(values_by_seed):
    """{name: [metric per seed]} -> {name: 'mean +- std'} strings for the report."""
    out = {}
    for k, v in values_by_seed.items():
        v = np.asarray(v, np.float64)
        out[k] = f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v.mean():.1f}"
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (for many pairwise comparisons in the cross-eval matrix)."""
    p = np.asarray(pvals, np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def bootstrap_p(a, b, n=10000, seed=0):
    """Two-sided p-value for mean(a - b) = 0 by sign-flip permutation on paired differences."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(1))
    return float((np.sum(null >= obs) + 1) / (n + 1))
