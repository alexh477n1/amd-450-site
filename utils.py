import pandas as pd
import numpy as np
from scipy import stats, signal
from sklearn.mixture import GaussianMixture
from typing import Dict, Optional, List, Any
import streamlit as st

# CONFIGURATION
CONFIG = {
    # General settings
    'slow_window': 30,
    'fast_window': 30,
    'min_series_len': 15,
    'stats_head_tail_len': 200,
    'kde_points': 200,

    # Detection tuning
    'peak_prominence': 0.2,
    'look_forward_window': 100,
    'min_persistence_ratio': 0.7,
    'sigma_head': 0.4,
    'min_rel_shift': 0.005,

    # Truncation tuning
    'trunc_ratio_threshold': 0.25,
    'trunc_skew_threshold': 1.0,

    # IQR tuning
    'floor_quantile': 0.0000001,
    'ceiling_quantile': 0.9999999,
    'tight_iqr_scale': 0.3,

    # Visualization
    'time_slice_window': 150,
    'rolling_mean_window': 150,

    # Mode detection
    'mode_window': 100,
    'mode_consistency': 15,

    # Rolling GMM Split
    'gmm_window': 50,

    # Constants
    'gmm_strictness': 1.5,
    'alpha': 0.05,
    'perm_resamples': 999,

    # Weights for each statistical tests
    'weights': {
        'Brunner-Munzel': 2.0,
        'Mann-Whitney U': 0.5,
        'Permutation': 2.0,
        'Kolmogorov-Smirnov': 0.5,
        'Cramer-von Mises': 0.5,
        'Anderson-Darling': 0.5,
        'Epps-Singleton': 1.0
    }
}

# SHARED HELPER FUNCTIONS


@st.cache_data
def get_distribution_modality(series: pd.Series) -> str:
    clean_vals = series.dropna()
    if clean_vals.nunique() <= 1 or clean_vals.std() < 1e-6:
        return "Unimodal"

    try:
        if len(clean_vals) > 2000:
            clean_vals = clean_vals.sample(2000, random_state=42)

        kde = stats.gaussian_kde(clean_vals)
        x_grid = np.linspace(
            clean_vals.min(), clean_vals.max(), CONFIG['kde_points'])
        kde_values = kde(x_grid)

        peaks, _ = signal.find_peaks(kde_values)
        significant_peaks = [
            p for p in kde_values[peaks]
            if p > CONFIG['peak_prominence'] * kde_values.max()
        ]
        return "Bimodal" if len(significant_peaks) >= 2 else "Unimodal"
    except Exception:
        return "Unimodal"


def calculate_change_metrics(data: pd.Series, start_idx: int, end_idx: int) -> Dict:
    if start_idx == 0 and end_idx == 0:
        return {
            'abs_delta': 0.0, 'pct_change': 0.0,
            'mean_cold': 0.0, 'mean_stable': 0.0,
            't_stat': 0.0, 'p_val': 1.0,
            'warmup_rate': 0.0
        }

    cold_slice = data.iloc[:start_idx] if start_idx > 0 else data.iloc[:10]
    stable_slice = data.iloc[end_idx:]
    if stable_slice.empty:
        stable_slice = data.iloc[-10:]

    mean_cold = cold_slice.mean()
    mean_stable = stable_slice.mean()
    abs_delta = mean_stable - mean_cold
    pct_change = ((mean_stable - mean_cold) / mean_cold) * \
        100 if mean_cold != 0 else 0.0

    duration = max(1, end_idx - start_idx)
    warmup_rate = abs(abs_delta) / duration

    if len(cold_slice) > 2 and len(stable_slice) > 2:
        try:
            t_stat, p_val = stats.ttest_ind(
                stable_slice.dropna(),
                cold_slice.dropna(),
                equal_var=False,
                nan_policy='omit'
            )
        except Exception:
            t_stat, p_val = 0.0, 1.0
    else:
        t_stat, p_val = 0.0, 1.0

    return {
        'abs_delta': abs_delta,
        'pct_change': pct_change,
        'mean_cold': mean_cold,
        'mean_stable': mean_stable,
        't_stat': t_stat,
        'p_val': p_val,
        'warmup_rate': warmup_rate
    }


def get_rolling_gmm_labels(data: pd.Series, window_size: int = 50) -> pd.Series:
    clean_data = data.dropna()
    if len(clean_data) < window_size:
        return pd.Series(0, index=data.index)
    global_labels = np.full(len(data), -1)
    global_std = clean_data.std()
    step = window_size

    for i in range(0, len(data), step):
        end_i = min(i + window_size, len(data))
        if end_i - i < 20:
            continue
        chunk = data.iloc[i:end_i]
        chunk_clean = chunk.dropna()
        if chunk_clean.empty or chunk_clean.nunique() < 5:
            label = 1 if chunk_clean.median() < clean_data.median() else 0
            global_labels[i:end_i] = label
            continue
        X = chunk_clean.values.reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=1)
            gmm.fit(X)
            means = gmm.means_.flatten()
            if abs(means[0] - means[1]) < (global_std * 0.25):
                local_preds = np.full(len(chunk), 0)
            else:
                local_preds = gmm.predict(chunk.fillna(
                    chunk.mean()).values.reshape(-1, 1))
                if means[0] < means[1]:
                    local_preds = 1 - local_preds
            global_labels[i:end_i] = local_preds
        except Exception:
            global_labels[i:end_i] = 0

    res_series = pd.Series(global_labels, index=data.index)
    res_series = res_series.replace(-1,
                                    np.nan).ffill().bfill().fillna(0).astype(int)
    return res_series


def calculate_warmup_stats(data, idx):
    if idx == 0:
        return None
    warm = data.iloc[:idx].mean()
    steady = data.iloc[idx:].mean()
    delta_pct = ((steady - warm) / warm) * 100
    return {'mean_warm': warm, 'mean_steady': steady, 'delta_pct': delta_pct}


def check_modality(chunk_data):
    """Checks for Multimodality using GMM + Constant Sigma Separation."""
    sigma_separation = CONFIG['gmm_strictness']
    X = chunk_data.values.reshape(-1, 1)
    try:
        gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
        gmm2 = GaussianMixture(n_components=2, random_state=42).fit(X)

        if gmm2.bic(X) < gmm1.bic(X) - 20:
            means = gmm2.means_.flatten()
            stds = np.sqrt(gmm2.covariances_.flatten())
            peak_dist = abs(means[0] - means[1])
            combined_spread = stds[0] + stds[1]

            if peak_dist > (sigma_separation * combined_spread):
                return True, means
        return False, gmm1.means_.flatten()
    except:
        return False, [chunk_data.mean()]


def remove_outliers_robust(series_data, k=7.0):
    """Median Absolute Deviation method."""
    if len(series_data) == 0:
        return series_data
    vals = series_data.astype(float)
    m = vals.median()
    mad = (vals - m).abs().median()
    if mad == 0:
        return series_data
    lo = m - k * mad
    hi = m + k * mad
    return series_data[(vals >= lo) & (vals <= hi)]


def diff_means_statistic(x, y):
    """Statistic function for Permutation Test."""
    return np.mean(x) - np.mean(y)


def get_hodges_lehmann_delta(x, y):
    """
    Calculates the Hodges-Lehmann estimator for the shift between two samples.
    Shift = Median(x[i] - y[j]) for all pairs.
    Optimized with sampling for speed if data is large.
    """

    n_x, n_y = len(x), len(y)

    # Safety check for memory (N*M can explode)
    MAX_SAMPLES = 2000

    if n_x > MAX_SAMPLES:
        x = np.random.choice(x, MAX_SAMPLES, replace=False)
    if n_y > MAX_SAMPLES:
        y = np.random.choice(y, MAX_SAMPLES, replace=False)

    # Broadcasting to get pairwise differences: x - y
    diffs = x[:, np.newaxis] - y
    return np.median(diffs)
