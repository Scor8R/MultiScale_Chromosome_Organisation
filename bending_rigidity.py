#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress
import warnings
from math import isnan

# =========================================================
# User parameters
# =========================================================
files = {
    "GENENAME":("CORDINATE_FILE",NUMBEROFPARTICLE)
}

# Bead typing (from your model)
bps_large = 147.0
bps_small = 7.35
target_bps = 200.0
large_bead_radius = 5.0
small_bead_radius = 1.25
rad_tol = 0.3

# Physics
kB_T = 4.1  # pN·nm at room temp

# Contact probability settings
cutoff_nm = 25.0
max_s_fit = 100
s_min_fit = 3

# Correlation / lag-wise settings
MAX_LAG = 20
C_EPS = 1e-6       # ignore non-positive C(s)
S_MIN = 1          # min lag to include in lag-wise estimator
S_MAX = 15         # max lag to include (avoid very noisy tails)

# ---- Robust global Kb (tangent–tangent) settings ----
MAX_LAG_GLOBAL = 15       # column cap for the global method
S_MIN_GLOBAL   = 1        # fit start lag
S_MAX_GLOBAL   = 12       # fit end lag (short & reliable)
WEIGHT_GAMMA   = 1.5      # weights w = 1 / s^gamma
ROW_R2_MIN     = 0.20     # minimum weighted R^2 to consider
ROW_KEEP_Q     = 0.80     # keep top 80% rows by R^2 (trim worst ~20%)

# ---- Alpha fit & calibration settings ----
ALPHA_SMIN = 3                 # fit on mid-range s to avoid small-s artifacts
ALPHA_SMAX = 50                # cap top end to avoid noisy tails
ALPHA_WEIGHT_GAMMA = 0.0       # weights in log–log fit: w = 1/s^gamma
ALPHA_CALIBRATE_TO = "lagwise" # {"lagwise", "global", None}

# Robust aggregation & bootstrap
TRIM_PROP_LAG  = 0.10     # trim per-snapshot across lags (10% each tail)
TRIM_PROP_BOOT = 0.10     # trim across bootstrap replicates
NBOOT_LAGWISE  = 1000     # lag-wise trimmed mean bootstrap
NBOOT_ALPHA    = 2000     # alpha→Kb bootstrap
NBOOT_GLOBAL   = 2000     # global corr→Kb bootstrap
RNG = np.random.default_rng(42)

# =========================================================
# Helpers
# =========================================================
def bead_bp_count(radius):
    if abs(radius - large_bead_radius) <= rad_tol:
        return bps_large
    if abs(radius - small_bead_radius) <= rad_tol:
        return bps_small
    return bps_small if abs(radius - small_bead_radius) < abs(radius - large_bead_radius) else bps_large

def coarse_grain_chain(coords, radii, target_bps=200.0):
    bp_counts = np.array([bead_bp_count(r) for r in radii])
    cg_coords = []
    i = 0
    N = len(coords)
    while i < N:
        bp_sum = 0.0
        group_coords = []
        group_weights = []
        while i < N and bp_sum < target_bps:
            bp = bp_counts[i]
            bp_sum += bp
            group_coords.append(coords[i])
            group_weights.append(bp)
            i += 1
        if group_coords:
            group_coords = np.array(group_coords)
            weights = np.array(group_weights)
            cm = np.average(group_coords, axis=0, weights=weights)
            cg_coords.append(cm)
    if not cg_coords:
        return np.zeros((0,3))
    return np.vstack(cg_coords)

def estimate_effective_bead_length_nm(cg_snaps):
    """Per-locus contour length per CG bead from geometry: median |Δr| between consecutive CG beads."""
    dists = []
    for coords in cg_snaps:
        if coords.shape[0] < 2:
            continue
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        if seg.size:
            dists.append(seg)
    if not dists:
        return np.nan
    dists = np.concatenate(dists)
    return float(np.median(dists))

# ---------- Contact decay & alpha ----------
def compute_contact_decay(cg_snaps, cutoff=cutoff_nm, max_s=None):
    if len(cg_snaps) == 0:
        return np.array([]), np.array([])
    minN = min(arr.shape[0] for arr in cg_snaps)
    if max_s is None:
        max_s = min(minN-1, max_s_fit)
    max_s = min(max_s, minN-1)
    s_vals = np.arange(1, max_s+1)
    P = np.zeros_like(s_vals, dtype=float)
    counts = np.zeros_like(s_vals, dtype=int)
    for coords in cg_snaps:
        N = coords.shape[0]
        if N < 2:
            continue
        dmat = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)
        for idx, s in enumerate(s_vals):
            if s >= N:
                continue
            pairs = np.arange(0, N - s)
            d = dmat[pairs, pairs+s]
            P[idx] += np.mean(d <= cutoff)
            counts[idx] += 1
    mask = counts > 0
    P[mask] = P[mask] / counts[mask]
    P[~mask] = np.nan
    return s_vals, P

def fit_alpha_weighted(s_vals, P_vals, smin=ALPHA_SMIN, smax=ALPHA_SMAX, gamma=ALPHA_WEIGHT_GAMMA):
    """Weighted log–log fit: log P = a - alpha * log s, weights w = 1/s^gamma."""
    if smax is None:
        smax = np.max(s_vals) if len(s_vals) else None
    if smax is None:
        return np.nan, None
    m = (s_vals >= smin) & (s_vals <= smax) & np.isfinite(P_vals) & (P_vals > 0)
    if m.sum() < 3:
        return np.nan, None
    xs = np.log(s_vals[m].astype(float))
    ys = np.log(P_vals[m].astype(float))
    w  = 1.0 / np.power(s_vals[m].astype(float), gamma)
    b, a = np.polyfit(xs, ys, 1, w=w)  # ys ≈ a + b*xs, so alpha = -b
    alpha = -float(b)
    return alpha, {"slope": b, "intercept": float(a), "n": int(m.sum())}

def alpha_to_lp_nm(alpha, bead_length_nm):
    """Placeholder α→ℓp mapping in bead units, then multiply by per-locus effective bead length."""
    lp_beads = (3.0 - alpha) * 5.0
    return lp_beads * bead_length_nm

def contact_decay_per_snapshot(coords, cutoff, max_s_fit):
    """Compute P(s) for a single CG chain (one snapshot)."""
    N = coords.shape[0]
    if N < 3:
        return np.array([]), np.array([])
    max_s = min(max_s_fit, N-1)
    s_vals = np.arange(1, max_s+1)
    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    P = np.zeros_like(s_vals, dtype=float)
    for i, s in enumerate(s_vals):
        pairs = np.arange(0, N - s)
        d = dmat[pairs, pairs + s]
        P[i] = np.mean(d <= cutoff)
    return s_vals, P

def build_Ps_stack_for_locus(cg_snaps, cutoff, max_s_fit):
    """Return (s_vals_common, P_stack) with shape [M, len(s)], aligned across snapshots."""
    if len(cg_snaps) == 0:
        return np.array([]), np.zeros((0,0))
    Ps, lengths = [], []
    for coords in cg_snaps:
        s_vals, P = contact_decay_per_snapshot(coords, cutoff, max_s_fit)
        if s_vals.size > 0:
            Ps.append(P); lengths.append(s_vals.size)
    if not Ps:
        return np.array([]), np.zeros((0,0))
    Lmin = min(lengths)
    s_common = np.arange(1, Lmin+1)
    P_stack = np.zeros((len(Ps), Lmin), dtype=float)
    for k, P in enumerate(Ps):
        P_stack[k, :] = P[:Lmin]
    return s_common, P_stack

# ---------- Tangent correlation ----------
def unit_tangents(chain):
    T = np.diff(chain, axis=0)
    norms = np.linalg.norm(T, axis=1)
    nz = norms > 0
    T[nz] = T[nz] / norms[nz][:, None]
    T[~nz] = 0.0
    return T

def corr_curve_one_snapshot(chain, max_lag=MAX_LAG):
    N = chain.shape[0]
    if N < 5:
        return np.full(max_lag, np.nan)
    Smax = min(max_lag, N-2)
    T = unit_tangents(chain)
    C = np.full(max_lag, np.nan)
    for s in range(1, Smax+1):
        v1 = T[:-s]; v2 = T[s:]
        if v1.shape[0] == 0: continue
        dots = np.sum(v1 * v2, axis=1)
        if dots.size > 0: C[s-1] = np.mean(dots)
    return C

def Cmat_per_snapshot(cg_snaps, max_lag=MAX_LAG):
    if len(cg_snaps) == 0:
        return np.zeros((0, max_lag)), np.arange(1, max_lag+1)
    M = len(cg_snaps)
    Cmat = np.full((M, max_lag), np.nan, dtype=float)
    for k, coords in enumerate(cg_snaps):
        Cmat[k, :] = corr_curve_one_snapshot(coords, max_lag=max_lag)
    s_vals = np.arange(1, max_lag+1)
    return Cmat, s_vals

def fit_alpha_unweighted(s_vals, P_vals, smin=ALPHA_SMIN, smax=ALPHA_SMAX):
    """Unweighted log–log fit: log P = a - alpha * log s (mimics final_rigidity.py)."""
    if s_vals.size == 0 or P_vals.size == 0:
        return np.nan, None
    smax = min(smax, int(s_vals.max()))
    m = (s_vals >= smin) & (s_vals <= smax) & np.isfinite(P_vals) & (P_vals > 0)
    if m.sum() < 3:
        return np.nan, None
    x = np.log(s_vals[m].astype(float))
    y = np.log(P_vals[m].astype(float))
    slope, intercept, r, p, se = linregress(x, y)  # unweighted
    alpha = -float(slope)
    return alpha, {"slope": float(slope), "intercept": float(intercept), "n": int(m.sum())}

# ---- Weighted/robust global fit utilities ----
def _weighted_linfit(x, y, w):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < 3: return np.nan, np.nan
    b, a = np.polyfit(x[m], y[m], 1, w=w[m])  # slope, intercept
    return float(b), float(a)

def _weighted_r2(y, yhat, w):
    m = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0)
    if m.sum() < 3: return np.nan
    y, yhat, w = y[m], yhat[m], w[m]
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w*(y - yhat)**2)
    ss_tot = np.sum(w*(y - ybar)**2)
    if ss_tot <= 0: return np.nan
    return float(1.0 - ss_res/ss_tot)

def _lp_from_meanC_weighted(C_mean, s_vals, smin=S_MIN_GLOBAL, smax=S_MAX_GLOBAL, eps=C_EPS, gamma=WEIGHT_GAMMA):
    if smax is None: smax = s_vals.max()
    m = (s_vals >= smin) & (s_vals <= smax) & np.isfinite(C_mean) & (C_mean > eps)
    if m.sum() < 3: return np.nan, np.nan
    x = s_vals[m].astype(float)
    y = np.log(C_mean[m])
    w = 1.0 / np.power(x, gamma)
    b, a = _weighted_linfit(x, y, w)
    if not np.isfinite(b) or b >= 0: return np.nan, np.nan
    yhat = a + b*x
    r2 = _weighted_r2(y, yhat, w)
    lp_beads = -1.0 / b
    return float(lp_beads), r2

def _row_quality_weighted(C_row, s_vals):
    lp, r2 = _lp_from_meanC_weighted(C_row, s_vals)
    return -np.inf if not np.isfinite(r2) else r2

def _filter_rows_by_quality(Cmat, s_vals, r2_min=ROW_R2_MIN, keep_q=ROW_KEEP_Q):
    if Cmat.size == 0: return Cmat
    r2s = np.array([_row_quality_weighted(Cmat[i, :], s_vals) for i in range(Cmat.shape[0])], float)
    keep = r2s >= r2_min
    if keep.sum() > 0 and np.isfinite(r2s[keep]).any():
        thr = np.quantile(r2s[keep], keep_q)
        keep &= (r2s >= thr)
    else:
        finite = np.isfinite(r2s)
        thr = np.quantile(r2s[finite], keep_q) if finite.any() else -np.inf
        keep = (r2s >= thr)
    return Cmat[keep, :]

# ---------- Robust summaries / bootstrap ----------
def trimmed_summary(x, trim=TRIM_PROP_BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan, np.nan, np.nan, 0
    x = np.sort(x)
    k = int(np.floor(trim * x.size))
    core = x[k: x.size-k] if 2*k < x.size else x
    center = float(np.mean(core))
    lo, hi = np.percentile(core, [2.5, 97.5]).astype(float)
    return center, lo, hi, core.size

def bootstrap_over_snapshots(values, nboot, reducer, rng=RNG):
    values = [v for v in values if v is not None]
    M = len(values)
    if M == 0: return np.array([])
    out = np.empty(nboot, dtype=float)
    for b in range(nboot):
        idx = rng.integers(0, M, size=M)
        sample = [values[i] for i in idx]
        out[b] = float(reducer(sample))
    return out

# --- Lag-wise estimator (per-snapshot Kb) ---
def lp_lagwise_per_snapshot(C_row, s_vals, smin=S_MIN, smax=S_MAX, trim=TRIM_PROP_LAG, eps=C_EPS):
    if smax is None: smax = s_vals.max()
    mask = np.isfinite(C_row) & (s_vals >= smin) & (s_vals <= smax) & (C_row > eps) & (C_row < 1.0)
    if mask.sum() < 3: return np.nan
    lp_s = -s_vals[mask] / np.log(C_row[mask])  # beads
    lp_s = np.sort(lp_s)
    k = int(np.floor(trim * lp_s.size))
    core = lp_s[k: lp_s.size-k] if 2*k < lp_s.size else lp_s
    return float(np.median(core))  # beads

def kb_from_snapshots(Cmat, s_vals, bead_len_nm):
    kb = []
    for row in Cmat:
        lp_beads = lp_lagwise_per_snapshot(row, s_vals)
        if np.isfinite(lp_beads):
            lp_nm = lp_beads * bead_len_nm
            kb.append(kB_T * lp_nm)
    return np.array(kb, dtype=float)

# =========================================================
# Main
# =========================================================
all_cg_snaps = {}
print("Building coarse-grained snapshots...")
for name, (path, beads) in files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input: {path}")
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["x","y","z","radius"])
    raw["snap"] = raw.index // beads
    cg_snaps = []
    for _, snap_df in raw.groupby("snap"):
        coords = snap_df[["x","y","z"]].values
        radii  = snap_df["radius"].values
        cg = coarse_grain_chain(coords, radii, target_bps=target_bps)
        if cg.shape[0] >= 5:
            cg_snaps.append(cg)
    all_cg_snaps[name] = cg_snaps
    mean_len = np.mean([s.shape[0] for s in cg_snaps]) if cg_snaps else np.nan
    print(f"  {name:8s}: {len(cg_snaps)} snaps, mean length ~ {mean_len:.1f} beads")

print("\nEstimating effective bead length per locus (nm) from geometry...")
eff_bead_len_nm = {}
for name, snaps in all_cg_snaps.items():
    eff_bead_len_nm[name] = estimate_effective_bead_length_nm(snaps)
    print(f"  {name:8s}: effective bead length ≈ {eff_bead_len_nm[name]:.2f} nm")

print("\nPrecomputing per-snapshot P(s) stacks for alpha mapping...")
alpha_precomp = {}
for name, snaps in all_cg_snaps.items():
    s_comm, P_stack = build_Ps_stack_for_locus(snaps, cutoff_nm, max_s_fit)
    alpha_precomp[name] = (s_comm, P_stack)
    print(f"  {name:8s}: P(s) stack {P_stack.shape}")

print("\nPrecomputing C(s) per snapshot...")
precomp = {}
for name, snaps in all_cg_snaps.items():
    Cmat, s_vals = Cmat_per_snapshot(snaps, max_lag=MAX_LAG)
    precomp[name] = (Cmat, s_vals)
    print(f"  {name:8s}: C(s) matrix {Cmat.shape}")

# ---------------------------------------------------
# A) Lag-wise WLC estimator → Kb per snapshot → CI
# ---------------------------------------------------
print("\nLag-wise WLC estimator → one Kb per snapshot → robust bootstrap across snapshots")
Kb_lag_stats = {}
Kb_lag_boots = {}
for name in files.keys():
    Cmat, s_vals = precomp[name]
    b_len = eff_bead_len_nm[name]
    kb_snap = kb_from_snapshots(Cmat, s_vals, b_len)

    def reducer(samples):
        return trimmed_summary(np.array(samples), trim=TRIM_PROP_BOOT)[0]

    kb_boot = bootstrap_over_snapshots(kb_snap, NBOOT_LAGWISE, reducer, rng=RNG)
    Kb_lag_boots[name] = kb_boot
    if kb_boot.size == 0:
        Kb_lag_stats[name] = (np.nan, np.nan, np.nan, 0)
        continue
    center, lo, hi, _ = trimmed_summary(kb_boot, trim=TRIM_PROP_BOOT)
    Kb_lag_stats[name] = (center, lo, hi, kb_snap.size)
    print(f"  {name:8s}: Kb_lagwise = {center:.1f} pN·nm², 95% CI=({lo:.1f}, {hi:.1f}); n_snaps={kb_snap.size}")

# ---------------------------------------------------
# B) Global tangent–tangent correlation → Kb (ensemble, robust)
# ---------------------------------------------------
print("\nGlobal tangent–tangent correlation → Kb (ensemble-fit) with bootstrap CI (robust)")
Kb_global_stats = {}
Kb_global_boots = {}

for name, snaps in all_cg_snaps.items():
    b_len = eff_bead_len_nm[name]
    Cmat_raw, s_vals_full = precomp[name]
    if Cmat_raw.size == 0 or not np.isfinite(b_len):
        Kb_global_stats[name] = (np.nan, np.nan, np.nan, 0)
        print(f"  {name:8s}: insufficient data for global Kb")
        continue

    # 1) restrict lag columns
    Smax = min(MAX_LAG_GLOBAL, Cmat_raw.shape[1])
    s_vals = s_vals_full[:Smax]
    Cmat = Cmat_raw[:, :Smax]

    # 2) filter snapshot rows by weighted-R^2 on the short window
    Cmat_filt = _filter_rows_by_quality(Cmat, s_vals, r2_min=ROW_R2_MIN, keep_q=ROW_KEEP_Q)
    n_rows_kept = Cmat_filt.shape[0]

    def reducer_global(rows_as_arrays):
        if len(rows_as_arrays) == 0:
            return np.nan
        C_mean = np.nanmean(np.vstack(rows_as_arrays), axis=0)
        lp_beads, r2 = _lp_from_meanC_weighted(C_mean, s_vals,
                                               smin=S_MIN_GLOBAL, smax=S_MAX_GLOBAL,
                                               eps=C_EPS, gamma=WEIGHT_GAMMA)
        if not np.isfinite(lp_beads):
            return np.nan
        return kB_T * (lp_beads * b_len)

    rows_list = [Cmat_filt[i, :] for i in range(n_rows_kept)]
    kb_boot = bootstrap_over_snapshots(rows_list, NBOOT_GLOBAL, reducer_global, rng=RNG)
    Kb_global_boots[name] = kb_boot

    if kb_boot.size == 0:
        Kb_global_stats[name] = (np.nan, np.nan, np.nan, 0)
        print(f"  {name:8s}: insufficient data for robust global Kb")
        continue

    center, lo, hi, _ = trimmed_summary(kb_boot, trim=TRIM_PROP_BOOT)
    Kb_global_stats[name] = (center, lo, hi, n_rows_kept)
    print(f"  {name:8s}: Kb_global  = {center:.1f} pN·nm², 95% CI=({lo:.1f}, {hi:.1f}); kept n={n_rows_kept}/{Cmat.shape[0]}")

# ---------------------------------------------------
# C) Contact-decay α mapping → Kb(α) + CI (robust & calibrated)
# ---------------------------------------------------
print("\nContact-decay α mapping → Kb(α) with bootstrap CI (unweighted fit, optional calibration)")
Kb_alpha_stats_raw = {}
Kb_alpha_boots_raw = {}

for name, snaps in all_cg_snaps.items():
    b_len = eff_bead_len_nm[name]
    s_comm, P_stack = alpha_precomp[name]   # per-snapshot aligned P(s)

    def alpha_from_Pmean_unweighted(P_mean, s_vals):
        if P_mean.size == 0:
            return np.nan
        a, _ = fit_alpha_unweighted(s_vals, P_mean,
                                    smin=ALPHA_SMIN, smax=min(ALPHA_SMAX, s_vals.max()))
        return a

    def reducer_alpha(sample_rows):
        # sample_rows is a list of P(s) rows; mimic pooled final_rigidity: average P(s), then fit
        if len(sample_rows) == 0:
            return np.nan
        P_mean = np.nanmean(np.vstack(sample_rows), axis=0)
        a = alpha_from_Pmean_unweighted(P_mean, s_comm)
        if not np.isfinite(a):
            return np.nan
        # raw (uncalibrated) placeholder mapping
        return kB_T * alpha_to_lp_nm(a, b_len)
    
    rows = [P_stack[i, :] for i in range(P_stack.shape[0])]
    kb_boot_raw = bootstrap_over_snapshots(rows, NBOOT_ALPHA, reducer_alpha, rng=RNG)
    Kb_alpha_boots_raw[name] = kb_boot_raw

    if kb_boot_raw.size > 0:
        center, lo, hi, _ = trimmed_summary(kb_boot_raw, trim=TRIM_PROP_BOOT)
    else:
        center = lo = hi = np.nan

    Kb_alpha_stats_raw[name] = (center, lo, hi, len(rows))
    print(f"  {name:8s}: Kb_alpha(raw) = {center:.1f} pN·nm², 95% CI=({lo:.1f}, {hi:.1f}); n_snaps={len(rows)}")

# ----- Calibrate α→Kb scale so means match reference (lagwise or global) -----
Kb_alpha_stats = {}
Kb_alpha_boots = {}
scale_ref = 1.0
if ALPHA_CALIBRATE_TO in {"lagwise", "global"}:
    ref = Kb_lag_stats if ALPHA_CALIBRATE_TO == "lagwise" else Kb_global_stats
    ratios = []
    for k in Kb_alpha_stats_raw.keys():
        m_ref = ref.get(k, (np.nan,))[0]
        m_raw = Kb_alpha_stats_raw.get(k, (np.nan,))[0]
        if np.isfinite(m_ref) and np.isfinite(m_raw) and m_raw > 0:
            ratios.append(m_ref / m_raw)
    if len(ratios) > 0:
        scale_ref = float(np.median(ratios))
    print(f"\nα→Kb calibration factor (to match {ALPHA_CALIBRATE_TO} means) = {scale_ref:.3f}")

for k, (m, lo, hi, n) in Kb_alpha_stats_raw.items():
    boots = Kb_alpha_boots_raw.get(k, np.array([]))
    boots_cal = boots * scale_ref if boots.size else boots
    Kb_alpha_boots[k] = boots_cal
    if boots_cal.size:
        mc, loc, hic, _ = trimmed_summary(boots_cal, trim=TRIM_PROP_BOOT)
        Kb_alpha_stats[k] = (mc, loc, hic, n)
    else:
        Kb_alpha_stats[k] = (np.nan, np.nan, np.nan, n)
    print(f"  {k:8s}: Kb_alpha(cal) = {Kb_alpha_stats[k][0]:.1f} pN·nm², 95% CI=({Kb_alpha_stats[k][1]:.1f}, {Kb_alpha_stats[k][2]:.1f})")

# =============================
# Save per-locus mechanics → kb_summary.csv
# =============================
def _fetch(stats_dict, name, idx, default=np.nan):
    try:
        v = stats_dict.get(name, (np.nan, np.nan, np.nan, np.nan))[idx]
        return float(v)
    except Exception:
        return float(default)
preferred_order = ["hoxa13", "hoxb4", "lin28a", "nanog"]
rows = []
for locus in preferred_order:
    row = {
        "system": locus,
        # main columns used downstream
        "Kb_lagwise":     _fetch(Kb_lag_stats,    locus, 0),
        "Kb_global":      _fetch(Kb_global_stats, locus, 0),
        "Kb_alpha_cal":   _fetch(Kb_alpha_stats,  locus, 0),  # calibrated α→Kb
        "bead_len_nm":    float(eff_bead_len_nm.get(locus, np.nan)),
        # optional transparency columns
        "Kb_lagwise_CI_lo": _fetch(Kb_lag_stats,    locus, 1),
        "Kb_lagwise_CI_hi": _fetch(Kb_lag_stats,    locus, 2),
        "Kb_lagwise_n":     _fetch(Kb_lag_stats,    locus, 3),
        "Kb_global_CI_lo":  _fetch(Kb_global_stats, locus, 1),
        "Kb_global_CI_hi":  _fetch(Kb_global_stats, locus, 2),
        "Kb_global_n":      _fetch(Kb_global_stats, locus, 3),
        "Kb_alpha_CI_lo":   _fetch(Kb_alpha_stats,  locus, 1),
        "Kb_alpha_CI_hi":   _fetch(Kb_alpha_stats,  locus, 2),
        "Kb_alpha_n":       _fetch(Kb_alpha_stats,  locus, 3),
    }
    # If you also want the *raw* (uncalibrated) alpha numbers, include them too:
    if "Kb_alpha_stats_raw" in globals():
        row["Kb_alpha_raw"]        = _fetch(Kb_alpha_stats_raw, locus, 0)
        row["Kb_alpha_raw_CI_lo"]  = _fetch(Kb_alpha_stats_raw, locus, 1)
        row["Kb_alpha_raw_CI_hi"]  = _fetch(Kb_alpha_stats_raw, locus, 2)
    rows.append(row)

kb_df = pd.DataFrame(rows)

print("\nkb_summary preview:")
print(kb_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


kb_df.to_csv("kb_summary.csv", index=False)
print("\nWrote kb_summary.csv")
# =========================================================
# Plots (3 forest plots, clean; no significance stars)
# =========================================================
def forest_plot(stats_dict, title, xlim=None, outfile=None):
    # name -> (mean, lo, hi, n)
    items = [(k,)+stats_dict[k] for k in stats_dict.keys()]
    items = sorted(items, key=lambda t: (t[1] if np.isfinite(t[1]) else -np.inf), reverse=True)
    labels = [t[0] for t in items]
    means  = np.array([t[1] for t in items], float)
    los    = np.array([t[2] for t in items], float)
    his    = np.array([t[3] for t in items], float)

    plt.rcParams.update({
        "font.size": 25,
        "axes.labelsize": 25,
        "axes.titlesize": 25,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
    })
    fig, ax = plt.subplots(figsize=(7, 7))
    y = np.arange(len(labels))

    xerr_lo = means - los
    xerr_hi = his - means
    ax.errorbar(means, y, xerr=[xerr_lo, xerr_hi], fmt="D", ms=5, lw=2.3,
                capsize=6, elinewidth=2.3, mew=0.0)

    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Bending rigidity $K_b$ (pN·nm$^2$)")
    #ax.set_title(title, pad=18)

    if xlim is None:
        xmin = np.nanmin(los) if np.isfinite(los).any() else np.nanmin(means)
        xmax = np.nanmax(his) if np.isfinite(his).any() else np.nanmax(means)
        rng = xmax - xmin
        xmin = max(0, xmin - 0.05*rng)
        xmax = xmax + 0.05*rng
    else:
        xmin, xmax = xlim
    ax.set_xlim(xmin, xmax)
    # Y-axis limits with padding
    ax.set_ylim(-1 , len(labels))
    # Keep ticks sparse and readable
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", which="both", linestyle=":", linewidth=0.8, alpha=0.6)

    for i, m in enumerate(means):
        if np.isfinite(m):
            ax.text(m, y[i]+0.18, f"{m:.1f}", ha="center", va="bottom", fontsize=25)

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=300)
    plt.show()

# 1) Lag-wise WLC Kb
forest_plot(Kb_lag_stats, "Lag-wise WLC: $K_b$ per locus (trimmed mean, 95% CI)",outfile="Bending_ridigity_plots/Lagwise_WLC_Kb.pdf")

# 2) Global tangent–tangent correlation Kb (robust)
forest_plot(Kb_global_stats, "Global tangent–tangent correlation: $K_b$ (95% CI)",outfile="Bending_ridigity_plots/Global_Tangent_Kb.pdf")

# 3) Alpha-mapped Kb (calibrated)
forest_plot(Kb_alpha_stats, "Contact-decay $\\alpha$ mapping (calibrated): $K_b$ (95% CI)",outfile="Bending_ridigity_plots/AlphaMapped_Kb.pdf")

# Optional: Contact decay P(s) plot (aggregate)
print("\nGenerating contact-decay P(s) plot...")
fig, ax = plt.subplots(figsize=(7,5))
for name in files.keys():
    s_vals, P_s = compute_contact_decay(all_cg_snaps[name], cutoff=cutoff_nm, max_s=max_s_fit)
    ax.scatter(s_vals, P_s, s=10, label=name)
    a, fit = fit_alpha_weighted(s_vals, P_s,
                                smin=ALPHA_SMIN, smax=min(ALPHA_SMAX, (s_vals.max() if len(s_vals) else ALPHA_SMAX)),
                                gamma=ALPHA_WEIGHT_GAMMA)
    if fit is not None:
        xs = np.linspace(max(2, s_vals[1] if len(s_vals)>1 else 2), min(ALPHA_SMAX, s_vals.max() if len(s_vals)>0 else ALPHA_SMAX), 60)
        # in log space, y = a + b*log(s); here b = -alpha
        ax.plot(xs, np.exp(fit["intercept"] + (-a)*np.log(xs)), linestyle='--')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel("contour separation s (CG beads)")
ax.set_ylabel("P(s) (contact probability)")
ax.set_title("Contact decay P(s) (log–log)")
ax.legend()
fig.tight_layout()
fig.savefig("Bending_ridigity_plots/contact_decay.pdf", dpi=300)
plt.show()

print("\nDone.")
