import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Optional, Tuple, Any
import zipfile
import io
import utils


# DETECTION HELPERS

def identify_dynamic_limit(segment: pd.Series, trunc_type: str) -> float:
    clean = segment.dropna()
    if clean.empty:
        return 0.0
    if trunc_type == 'floor':
        extreme_values = clean[clean <= clean.quantile(0.10)]
        return extreme_values.mode().iloc[0] if not extreme_values.empty else clean.min()
    elif trunc_type == 'ceiling':
        extreme_values = clean[clean >= clean.quantile(0.90)]
        return extreme_values.mode().iloc[0] if not extreme_values.empty else clean.max()
    return clean.median()


def get_tight_iqr_bounds(data_segment: pd.Series, scale: float = 0.3, trunc_type: str = 'none') -> Tuple[float, float]:
    if trunc_type != 'none':
        target_limit = identify_dynamic_limit(data_segment, trunc_type)
        local_noise = data_segment.std() * 0.1
        if trunc_type == 'floor':
            lower = target_limit - local_noise
            upper = target_limit + \
                max(local_noise * 2.0, data_segment.std() * 0.05)
        else:
            lower = target_limit - \
                max(local_noise * 2.0, data_segment.std() * 0.05)
            upper = target_limit + local_noise
        return lower, upper
    q25, q50, q75 = data_segment.quantile([0.25, 0.5, 0.75])
    width = (q75 - q25) * scale
    return q50 - (width / 2), q50 + (width / 2)


def analyze_truncation(segment: pd.Series) -> Dict[str, Any]:
    clean = segment.dropna()
    if len(clean) < 5:
        return {'type': 'none', 'ratio': 1.0, 'skew': 0.0}
    skew_val = stats.skew(clean)
    p05 = clean.quantile(0.05)
    p50 = clean.median()
    p95 = clean.quantile(0.95)
    bottom_spread = max(p50 - p05, 1e-9)
    top_spread = max(p95 - p50, 1e-9)
    floor_ratio = bottom_spread / top_spread
    ceiling_ratio = top_spread / bottom_spread
    ratio_thresh = utils.CONFIG['trunc_ratio_threshold']
    skew_thresh = utils.CONFIG['trunc_skew_threshold']

    if floor_ratio < ratio_thresh and skew_val > skew_thresh:
        return {'type': 'floor', 'ratio': floor_ratio, 'skew': skew_val}
    elif ceiling_ratio < ratio_thresh and skew_val < -skew_thresh:
        return {'type': 'ceiling', 'ratio': ceiling_ratio, 'skew': skew_val}
    display_ratio = min(floor_ratio, ceiling_ratio)
    return {'type': 'none', 'ratio': display_ratio, 'skew': skew_val}


@st.cache_data
def get_rolling_mode_trend(series: pd.Series, window: int = 40) -> Tuple[pd.Series, float]:
    clean = series.dropna()
    q25, q75 = clean.quantile([0.25, 0.75])
    iqr = q75 - q25
    if iqr == 0:
        bin_size = 0.1
    else:
        bin_size = max(iqr / 20.0, 0.01)
        if 0.08 < bin_size < 0.12:
            bin_size = 0.1
        elif 0.8 < bin_size < 1.2:
            bin_size = 1.0
    binned_series = (series / bin_size).round() * bin_size

    def calc_mode(x):
        m = stats.mode(x, keepdims=False)
        return m.mode if np.isscalar(m.mode) else m.mode[0]

    rolling_mode = binned_series.rolling(
        window=window, min_periods=1, center=True).apply(calc_mode, raw=True)
    return rolling_mode.bfill().ffill(), bin_size


@st.cache_data
def get_adaptive_smoothing(series: pd.Series) -> Tuple[pd.Series, pd.Series, str, Dict]:
    clean = series.dropna()
    modality = utils.get_distribution_modality(clean)
    eff_slow = max(min(utils.CONFIG['slow_window'], len(series) // 2), 5)
    fast_roller = series.rolling(
        window=utils.CONFIG['fast_window'], min_periods=1, center=True)
    slow_roller = series.rolling(window=eff_slow, min_periods=1, center=True)

    stats_report = {}
    limit = utils.CONFIG['stats_head_tail_len']
    head_stats = analyze_truncation(clean.iloc[:limit])
    tail_stats = analyze_truncation(clean.iloc[-limit:])
    stats_report['head'] = head_stats
    stats_report['tail'] = tail_stats
    trunc_type = tail_stats['type'] if tail_stats['type'] != 'none' else head_stats['type']

    if trunc_type == 'floor':
        fast_smoothed = fast_roller.quantile(utils.CONFIG['floor_quantile'])
        slow_smoothed = slow_roller.quantile(utils.CONFIG['floor_quantile'])
        method = f"Floor Track (p{int(utils.CONFIG['floor_quantile'] * 100)})"
        stats_report['status'] = "Truncated (Floor)"
    elif trunc_type == 'ceiling':
        fast_smoothed = fast_roller.quantile(utils.CONFIG['ceiling_quantile'])
        slow_smoothed = slow_roller.quantile(utils.CONFIG['ceiling_quantile'])
        method = f"Ceiling Track (p{int(utils.CONFIG['ceiling_quantile'] * 100)})"
        stats_report['status'] = "Truncated (Ceiling)"
    elif modality == "Bimodal":
        fast_smoothed = fast_roller.median()
        slow_smoothed = slow_roller.median()
        method = "Median (Bimodal)"
        stats_report['status'] = "Bimodal"
    else:
        fast_smoothed = fast_roller.median()
        slow_smoothed = slow_roller.median()
        method = "Median (Unimodal)"
        stats_report['status'] = "Normal"
    return fast_smoothed.bfill().ffill(), slow_smoothed.bfill().ffill(), method, stats_report


def check_look_forward_np(chunk_np: np.ndarray, limit: float, going_up: bool) -> bool:
    if len(chunk_np) == 0:
        return False
    # Use config variable instead of hardcoded 0.80
    threshold_ratio = utils.CONFIG['min_persistence_ratio']
    if going_up:
        return np.mean(chunk_np > limit) >= threshold_ratio
    return np.mean(chunk_np < limit) >= threshold_ratio


# CORE LOGIC: DETECT WARMUP

def detect_warmup(series: pd.Series, sensitivity: str = "auto") -> Optional[Dict]:
    if len(series) < utils.CONFIG['min_series_len']:
        return None
    clean_vals = series.dropna()
    fast_smooth, slow_smooth, smooth_method, stats_debug = get_adaptive_smoothing(
        series)
    h_type = stats_debug.get('head', {}).get('type', 'none')
    t_type = stats_debug.get('tail', {}).get('type', 'none')
    trunc_mode = t_type if t_type != 'none' else h_type
    modality = utils.get_distribution_modality(clean_vals)

    # BRANCH 1: Truncation (Uses Rolling Mode)
    if trunc_mode in ['floor', 'ceiling']:
        mode_trend, bin_size = get_rolling_mode_trend(
            series, window=utils.CONFIG['mode_window'])
        smooth_method = f"Rolling Mode (bin={bin_size:.2f})"
        limit = utils.CONFIG['stats_head_tail_len']
        head_mode = mode_trend.iloc[:limit].mode()[0]
        tail_part = clean_vals.iloc[-limit:]
        if trunc_mode == 'floor':
            tail_target = tail_part.quantile(0.01)
            end_tolerance = max(tail_part.std() * 0.2, bin_size)
        else:
            tail_target = tail_part.quantile(0.99)
            end_tolerance = max(tail_part.std() * 0.2, bin_size)
        mode_vals = mode_trend.values
        consistency = utils.CONFIG['mode_consistency']
        end_pos = 0
        search_start_end = limit // 2
        for i in range(search_start_end, len(mode_vals) - consistency):
            chunk = mode_vals[i: i + consistency]
            if trunc_mode == 'floor':
                is_at_end = np.all(chunk <= (tail_target + end_tolerance))
            else:
                is_at_end = np.all(chunk >= (tail_target - end_tolerance))
            if is_at_end:
                end_pos = i
                break
        start_pos = 0
        if end_pos > 0:
            for i in range(end_pos, 0, -1):
                if abs(mode_vals[i] - head_mode) < 1e-9:
                    start_pos = i + 1
                    break
        has_warmup = (start_pos > 0) and (
            end_pos > start_pos) and (end_pos - start_pos > 2)
        metrics = utils.calculate_change_metrics(series, start_pos, end_pos)
        return {
            'has_warmup': has_warmup, 'start_idx': start_pos, 'end_idx': end_pos,
            'start_label': series.index[start_pos] if has_warmup else None,
            'end_label': series.index[end_pos] if has_warmup else None,
            'warm_dur': end_pos - start_pos, 'method': smooth_method,
            'metrics': metrics, 'stats': stats_debug, 'trunc_mode': trunc_mode,
            'mode_info': {'head': head_mode, 'tail': tail_target, 'bin': bin_size}
        }

    # BRANCH 2: Normal / Persistence (Uses check_look_forward_np)
    else:
        limit = utils.CONFIG['stats_head_tail_len']
        if len(slow_smooth) < limit * 2:
            limit = len(slow_smooth) // 3
        head_part = clean_vals.iloc[:limit]
        tail_part = clean_vals.iloc[-limit:]
        h_tight_low, h_tight_high = get_tight_iqr_bounds(
            head_part, scale=utils.CONFIG['tight_iqr_scale'], trunc_type=trunc_mode)
        t_tight_low, t_tight_high = get_tight_iqr_bounds(
            tail_part, scale=utils.CONFIG['tight_iqr_scale'], trunc_type=trunc_mode)
        mu_head = fast_smooth.iloc[:limit].median()
        mu_tail = fast_smooth.iloc[-limit:].median()
        fast_np = fast_smooth.values
        slow_np = slow_smooth.values
        has_warmup, start_pos, end_pos = False, 0, 0
        idx_vals = np.arange(len(clean_vals))
        try:
            corr, _ = stats.spearmanr(clean_vals, idx_vals)
        except:
            corr = 0
        diff = abs(mu_tail - mu_head)
        shift_threshold = abs(mu_head * utils.CONFIG['min_rel_shift'])
        is_shifted = diff > shift_threshold

        if modality == "Bimodal" or abs(corr) > 0.10 or is_shifted:
            going_up = mu_tail > mu_head
            std_head = head_part.std()
            d_c = max(utils.CONFIG['sigma_head'] * std_head, shift_threshold)
            cold_lim = mu_head + d_c if going_up else mu_head - d_c
            look_fwd = utils.CONFIG['look_forward_window']
            curr = min(limit // 2, 5)
            max_pos = len(slow_np) - look_fwd
            loop_safety = 0
            while curr < max_pos:
                loop_safety += 1
                if loop_safety > 50:
                    break
                rough_start = -1
                for i in range(curr, max_pos):
                    chunk = slow_np[i: i + look_fwd]
                    if check_look_forward_np(chunk, cold_lim, going_up):
                        rough_start = i
                        break
                if rough_start == -1:
                    break
                precise_start = rough_start
                backtrack_limit = max(
                    0, rough_start - utils.CONFIG['slow_window'])
                for i in range(rough_start, backtrack_limit, -1):
                    val = fast_np[i]
                    is_inside = (val <= cold_lim) if going_up else (
                        val >= cold_lim)
                    if is_inside:
                        fwd_chunk = fast_np[i + 1: i + 1 + look_fwd]
                        if check_look_forward_np(fwd_chunk, cold_lim, going_up):
                            precise_start = i + 1
                            break
                fs = precise_start
                rough_end = -1
                target_min, target_max = t_tight_low, t_tight_high
                for i in range(fs + 1, max_pos):
                    val = fast_np[i]
                    is_at_limit = (val >= target_min and val <= target_max)
                    if is_at_limit:
                        chunk = fast_np[i: i +
                                        utils.CONFIG['look_forward_window']]
                        if len(chunk) == 0:
                            continue
                        ratio_inside = np.mean(
                            (chunk >= target_min) & (chunk <= target_max))
                        med = np.median(chunk)
                        if (ratio_inside > 0.60) or (med >= target_min and med <= target_max):
                            rough_end = i
                            break
                if rough_end == -1:
                    break
                fe = rough_end
                if fe > fs and (fe - fs) > 2:
                    has_warmup = True
                    start_pos, end_pos = fs, fe
                    break
                else:
                    curr = rough_end + 10
        if has_warmup and end_pos <= start_pos:
            has_warmup = False
        metrics = utils.calculate_change_metrics(series, start_pos, end_pos)
        return {
            'has_warmup': has_warmup, 'start_idx': start_pos, 'end_idx': end_pos,
            'start_label': fast_smooth.index[start_pos] if has_warmup else None,
            'end_label': fast_smooth.index[end_pos] if has_warmup else None,
            'warm_dur': end_pos - start_pos, 'method': smooth_method,
            'metrics': metrics, 'stats': stats_debug, 'trunc_mode': trunc_mode,
            'iqr_stats': {'head_low': h_tight_low, 'head_high': h_tight_high, 'tail_low': t_tight_low,
                          'tail_high': t_tight_high}
        }


# VISUALIZATION FUNCTIONS (WARMUP SPECIFIC)

def plot_time_slice_distributions(ax, data, window_size=150):
    clean_data = data.dropna()
    if clean_data.empty:
        return
    y_min, y_max = clean_data.min(), clean_data.max()
    x_grid = np.linspace(y_min, y_max, 100)
    num_slices = int(np.ceil(len(clean_data) / window_size))
    if num_slices > 20:
        window_size = len(clean_data) // 20
        num_slices = 20
    colors = plt.cm.viridis(np.linspace(0, 1, num_slices))
    max_dens = 0
    kdes = []
    for i in range(num_slices):
        start = i * window_size
        end = min((i + 1) * window_size, len(clean_data))
        slice_data = clean_data.iloc[start:end]
        if len(slice_data) < 2 or slice_data.std() == 0:
            kdes.append(None)
            continue
        try:
            kde = stats.gaussian_kde(slice_data)
            kde_vals = kde(x_grid)
            max_dens = max(max_dens, kde_vals.max())
            kdes.append((kde_vals, start, end))
        except:
            kdes.append(None)
    offset_step = max_dens * 0.5
    for i, item in enumerate(kdes):
        if item is None:
            continue
        kde_vals, start, end = item
        base_line = i * offset_step
        scaled_kde = base_line + kde_vals
        ax.fill_between(x_grid, base_line, scaled_kde,
                        color=colors[i], alpha=0.6)
        ax.plot(x_grid, scaled_kde, color='white', lw=1)
        ax.text(x_grid[-1], base_line + (max_dens * 0.1),
                f"Iter {start}-{end}", fontsize=8, va='center')
    ax.set_yticks([])
    ax.set_xlabel("Value Distribution")
    ax.set_title(
        f"Evolution of Distribution (Window ~{window_size})", fontsize=10)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_global_rolling_means(ax, data, g0, g1, info, window_size=150):
    ax.scatter(data.index, data, c='lightgray',
               s=5, alpha=0.3, label='Global Data')
    g0_roll = g0.rolling(window=window_size, min_periods=1).mean(
    ) if not g0.empty else pd.Series()
    g1_roll = g1.rolling(window=window_size, min_periods=1).mean(
    ) if not g1.empty else pd.Series()
    if not g0_roll.empty:
        ax.plot(g0_roll.index, g0_roll, color='darkred',
                lw=2.5, label='Group 0 (or Global) Mean')
    if not g1_roll.empty:
        ax.plot(g1_roll.index, g1_roll, color='navy',
                lw=2.5, label='Group 1 Rolling Mean')
    if info.get('has_warmup'):
        end_idx = info.get('end_label')
        try:
            ax.axvline(end_idx, color='purple', linestyle='--',
                       linewidth=2, label='Warmup End')
            ax.text(end_idx, data.max(), " End of Warmup", rotation=90, va='top', fontsize=10, fontweight='bold',
                    color='purple')
        except:
            pass
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Global Group Dynamics (Rolling Means)", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Value")


def plot_regime(ax, data, info, title, color_main, thresholds=None, overlays=None, group_labels=None, show_trend=True,
                show_iqr=False, simple_view=False):
    scatter_data = data.sample(5000).sort_index() if len(data) > 5000 else data
    s_size = 15 if simple_view else (2 if len(data) > 5000 else 5)

    if show_iqr and 'iqr_stats' in info and not simple_view and 'mode_info' not in info:
        qs = info['iqr_stats']
        for zone, col in [('head', 'skyblue'), ('tail', 'limegreen')]:
            if f'{zone}_low' in qs:
                ax.axhspan(qs[f'{zone}_low'], qs[f'{zone}_high'],
                           color=col, alpha=0.3, label=f'{zone.title()} Range')

    if 'mode_info' in info and not simple_view:
        m = info['mode_info']
        bin_h = m['bin'] / 2
        ax.axhspan(m['head'] - bin_h, m['head'] + bin_h,
                   color='skyblue', alpha=0.3, label='Head Mode')
        ax.axhspan(m['tail'] - bin_h, m['tail'] + bin_h,
                   color='limegreen', alpha=0.3, label='Tail (p1/p99)')

    if group_labels is not None:
        aligned_labels = group_labels.loc[scatter_data.index]
        g0_data = scatter_data[aligned_labels == 0]
        g1_data = scatter_data[aligned_labels == 1]
        if not g0_data.empty:
            ax.scatter(g0_data.index, g0_data, s=s_size, c='tomato', alpha=0.6,
                       label='Group 0 (Slow)')
        if not g1_data.empty:
            ax.scatter(g1_data.index, g1_data, s=s_size, c='dodgerblue', alpha=0.6,
                       label='Group 1 (Fast)')
    else:
        ax.scatter(scatter_data.index, scatter_data, s=s_size,
                   c='gray', alpha=0.5, label='Raw Data')

    if show_trend and not simple_view:
        if 'mode_info' in info:
            trend, _ = get_rolling_mode_trend(
                data, window=utils.CONFIG['mode_window'])
            method_lbl = "Rolling Mode"
        else:
            trend, _, _, _ = get_adaptive_smoothing(data)
            method_lbl = "Adaptive Median"
        ax.plot(trend.index, trend, color=color_main,
                lw=2.5, label=f'Trend ({method_lbl})')

    if thresholds is not None and not simple_view:
        ax.plot(data.index, thresholds, color='purple',
                linestyle='--', linewidth=2, label='Zonal Threshold')

    has_w = info.get('has_warmup', False)
    final_end_idx = 0
    if simple_view and overlays:
        for ov in overlays:
            if ov.get('end_label') is not None:
                final_end_idx = max(final_end_idx, ov['end_label'])
        if final_end_idx > 0:
            ax.axvline(final_end_idx, color='purple', linestyle='-',
                       linewidth=2, alpha=1.0, label='Warmup End')
            ax.text(final_end_idx, data.max(), " End of Warmup", rotation=90, va='top', fontsize=10, fontweight='bold',
                    color='purple')

    elif has_w:
        s_lbl = info.get('start_label')
        e_lbl = info.get('end_label')
        if s_lbl is not None and e_lbl is not None:
            if not show_iqr:
                ax.axvspan(
                    data.index[0], s_lbl, color='skyblue', alpha=0.1, label='Cold Zone')
                ax.axvspan(s_lbl, e_lbl, color='orange',
                           alpha=0.2, label='Warm-up')
                ax.axvspan(
                    e_lbl, data.index[-1], color='green', alpha=0.1, label='Stable')
            ax.axvline(s_lbl, color=color_main, linestyle='--', alpha=0.8)
            ax.axvline(e_lbl, color=color_main, linestyle='--', alpha=0.8)
            ax.axvline(e_lbl, color='purple', linestyle='-',
                       linewidth=2, alpha=1.0)
            ax.text(s_lbl, data.max(), f" Start", rotation=90, va='top', fontsize=8, color=color_main,
                    fontweight='bold')
            ax.text(e_lbl, data.max(), f" End", rotation=90, va='top',
                    fontsize=10, color='purple', fontweight='bold')

    ax.set_title(title, loc='left', fontweight='bold')
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Latency")


def plot_distribution(ax, data, title, color='purple'):
    if len(data) < 10 or data.nunique() <= 1:
        ax.hist(data, bins=5, color=color, alpha=0.5)
    else:
        sample = data.sample(2000) if len(data) > 2000 else data
        sns.kdeplot(sample, ax=ax, fill=True, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Value")
    ax.grid(True, alpha=0.3)


def render_analysis_tab(data_series: pd.Series, result_info: Dict, title: str, color: str, show_gmm: bool,
                        show_iqr: bool, group_labels=None, simple_view=False, overlays=None, subgroup_res=None):
    if result_info is None:
        st.info(f"Insufficient data for {title}.")
        return

    c1, c2, c3, c4 = st.columns(4)
    if result_info['has_warmup']:
        s_idx, e_idx = result_info['start_idx'], result_info['end_idx']
        s_lbl = result_info.get('start_label', s_idx)
        e_lbl = result_info.get('end_label', e_idx)

        def fmt_lbl(val):
            try:
                return f"{int(val)}"
            except:
                return f"{val}"

        time_to_start = data_series.iloc[:s_idx].sum()
        time_duration = data_series.iloc[s_idx:e_idx].sum()
        time_to_stable = data_series.iloc[:e_idx].sum()
        c1.metric("Warm-Up Start",
                  f"Idx {fmt_lbl(s_lbl)}", f"{time_to_start:.1f} ms")
        c2.metric("Warm-Up Duration",
                  f"{result_info['warm_dur']} iter", f"{time_duration:.1f} ms")
        c3.metric("Stable At", f"Idx {fmt_lbl(e_lbl)}",
                  f"{time_to_stable:.1f} ms (Total)")
    else:
        st.warning(f"No Warm-up Detected in {title}")
        c1.metric("Warm-Up Start", "-")
        c2.metric("Warm-Up Duration", "-")
        c3.metric("Stable At", "-")

    if title in ["Group 0", "Group 1"] and group_labels is not None:
        total_obs = len(group_labels)
        grp_val = 0 if title == "Group 0" else 1
        count = (group_labels == grp_val).sum()
        fraction = count / total_obs if total_obs > 0 else 0
        c4.metric("Group Fraction", f"{fraction:.1%}")
    else:
        c4.metric("Method Used", result_info['method'])

    # Weighted breakdown (specific to Global view with subgroups)
    show_weighted_breakdown = (title == "Global") and (
        subgroup_res is not None) and (group_labels is not None)
    if show_weighted_breakdown and subgroup_res:
        g0_res = subgroup_res.get('g0')
        g1_res = subgroup_res.get('g1')
        total_obs = len(data_series)
        count_0 = (group_labels == 0).sum()
        count_1 = (group_labels == 1).sum()
        frac_0 = count_0 / total_obs if total_obs > 0 else 0
        frac_1 = count_1 / total_obs if total_obs > 0 else 0
        d0 = g0_res.get('metrics', {}).get('abs_delta', 0) if g0_res else 0
        d1 = g1_res.get('metrics', {}).get('abs_delta', 0) if g1_res else 0
        p0 = g0_res.get('metrics', {}).get('pct_change', 0) if g0_res else 0
        p1 = g1_res.get('metrics', {}).get('pct_change', 0) if g1_res else 0
        weighted_avg = (d0 * frac_0) + (d1 * frac_1)
        weighted_pct = (p0 * frac_0) + (p1 * frac_1)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("G0 Abs Delta", f"{d0:.4f}",
                  f"{p0:.2f}% (Imp)", delta_color="off")
        m2.metric("G1 Abs Delta", f"{d1:.4f}",
                  f"{p1:.2f}% (Imp)", delta_color="off")
        m3.metric("Weighted Avg Change", f"{weighted_avg:.4f}",
                  f"{weighted_pct:.2f}% (W. Pct)", delta_color="off")
        m4.empty()

    elif result_info['has_warmup']:
        m = result_info.get('metrics', {})
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cold Avg", f"{m.get('mean_cold', 0):.4f}")
        s2.metric("Stable Avg", f"{m.get('mean_stable', 0):.4f}")
        s3.metric(
            "Abs Delta", f"{m.get('abs_delta', 0):.4f}", delta_color="off")
        s4.metric(
            "% Change", f"{m.get('pct_change', 0):.2f}%", delta_color="off")

    st.divider()

    if show_gmm:
        col_plot, col_dist = st.columns([3, 1])
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_regime(ax, data_series, result_info, f"{title} Timeline", color, show_iqr=show_iqr,
                    group_labels=group_labels, simple_view=simple_view, overlays=overlays)
        col_plot.pyplot(fig)
        with col_dist:
            limit = utils.CONFIG['stats_head_tail_len']
            fig_h, ax_h = plt.subplots(figsize=(3, 2))
            plot_distribution(
                ax_h, data_series.iloc[:limit], "Head Dist", color='skyblue')
            st.pyplot(fig_h)
            fig_t, ax_t = plt.subplots(figsize=(3, 2))
            plot_distribution(
                ax_t, data_series.iloc[-limit:], "Tail Dist", color='limegreen')
            st.pyplot(fig_t)
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_regime(ax, data_series, result_info, f"{title} Timeline", color, show_iqr=show_iqr,
                    group_labels=group_labels, simple_view=simple_view, overlays=overlays)
        st.pyplot(fig)


# MAIN TAB FUNCTIONALITY (WARMUP)

def render_warmup_tab(show_iqr, show_gmm):
    st.header("Single File / Batch Warmup Analysis")
    st.markdown(
        "Detailed breakdown of warmup phases, statistical trends, and distribution modality.")

    uploaded_files = st.file_uploader("Upload CSVs (Tab 1)", accept_multiple_files=True, type=['csv'],
                                      key="tab1_uploader")

    if not uploaded_files:
        st.info("Upload CSV files to begin Analysis.")
    else:
        file_map = {f.name: f for f in uploaded_files}
        file_list = list(file_map.keys())
        if 'file_selection' not in st.session_state or len(st.session_state.file_selection) != len(file_list):
            st.session_state.file_selection = pd.DataFrame(
                {"Select": [True] * len(file_list), "Filename": file_list})

        edited_df = st.data_editor(st.session_state.file_selection,
                                   column_config={"Select": st.column_config.CheckboxColumn(
                                       "Show", default=True)},
                                   disabled=["Filename"], hide_index=True, key="editor")
        selected_filenames = edited_df[edited_df["Select"]
                                       ]["Filename"].tolist()

        if selected_filenames:
            summary_data = []
            analysis_results = {}
            progress_bar = st.progress(0)

            for i, fname in enumerate(selected_filenames):
                file_obj = file_map[fname]
                file_obj.seek(0)
                df = pd.read_csv(file_obj)
                if df.empty:
                    continue
                data = pd.to_numeric(
                    df.iloc[:, 0], errors='coerce').dropna().reset_index(drop=True)

                modality = utils.get_distribution_modality(data)
                is_bimodal = False
                labels = pd.Series(0, index=data.index)

                if modality == "Bimodal":
                    labels = utils.get_rolling_gmm_labels(
                        data, window_size=utils.CONFIG['gmm_window'])
                    if labels.nunique() > 1:
                        is_bimodal = True
                    else:
                        labels = pd.Series(0, index=data.index)

                global_res = detect_warmup(data.copy(), "auto")

                # Code 1 Merging Logic
                g0 = data[labels == 0]
                g1 = data[labels == 1]
                g0_res, g1_res = None, None

                if is_bimodal:
                    g0_res = detect_warmup(g0, "group")
                    g1_res = detect_warmup(
                        g1, "group") if len(g1) > 50 else None

                    group_end_labels = []
                    group_start_labels = []
                    if g0_res and g0_res['has_warmup']:
                        group_end_labels.append(g0_res['end_label'])
                        group_start_labels.append(g0_res['start_label'])
                    if g1_res and g1_res['has_warmup']:
                        group_end_labels.append(g1_res['end_label'])
                        group_start_labels.append(g1_res['start_label'])

                    if group_end_labels:
                        final_end_label = max(group_end_labels)
                        final_start_label = min(group_start_labels)
                        try:
                            final_start_idx = data.index.get_loc(
                                final_start_label)
                            final_end_idx = data.index.get_loc(final_end_label)
                        except KeyError:
                            final_start_idx = 0
                            final_end_idx = 0

                        global_res['has_warmup'] = True
                        global_res['start_idx'] = final_start_idx
                        global_res['end_idx'] = final_end_idx
                        global_res['start_label'] = final_start_label
                        global_res['end_label'] = final_end_label
                        global_res['warm_dur'] = final_end_idx - \
                            final_start_idx
                        new_metrics = utils.calculate_change_metrics(
                            data, final_start_idx, final_end_idx)
                        global_res['metrics'] = new_metrics

                # FORCE STATS IF NO WARMUP
                if not global_res['has_warmup']:
                    limit = min(
                        utils.CONFIG['stats_head_tail_len'], len(data) // 3)
                    if limit > 5:
                        fallback_metrics = utils.calculate_change_metrics(
                            data, limit, len(data) - limit)
                        global_res['metrics'] = fallback_metrics

                dur_ms = 0
                if global_res['has_warmup']:
                    dur_ms = data.iloc[global_res['start_idx']:global_res['end_idx']].sum()

                def get_trunc_stats(res_obj, part):
                    if res_obj is None:
                        return 0.0, 0.0
                    return (
                        res_obj.get('stats', {}).get(
                            part, {}).get('ratio', 0.0),
                        res_obj.get('stats', {}).get(part, {}).get('skew', 0.0)
                    )

                gh_r, gh_s = get_trunc_stats(global_res, 'head')
                gt_r, gt_s = get_trunc_stats(global_res, 'tail')
                metrics = global_res.get('metrics', {})
                summary_data.append({
                    "No.": i + 1,
                    "Filename": fname,
                    "Detected": "✅" if global_res['has_warmup'] else "❌",
                    "Stable Idx": global_res.get('end_idx', '-') if global_res['has_warmup'] else "-",
                    "Duration (ms)": round(dur_ms, 2),
                    "Abs Delta": round(metrics.get('abs_delta', 0), 4),
                    "% Change": round(metrics.get('pct_change', 0), 2),
                    "T-Stat": round(metrics.get('t_stat', 0), 4),
                    "P-Value": round(metrics.get('p_val', 1.0), 4),
                    "Modality": modality,
                    "H.Ratio": round(gh_r, 2),
                    "H.Skew": round(gh_s, 2),
                    "T.Ratio": round(gt_r, 2),
                    "T.Skew": round(gt_s, 2),
                    "Trunc Mode": global_res.get('trunc_mode', 'none')
                })

                analysis_results[fname] = {
                    'data': data, 'global': global_res, 'g0': g0_res, 'g1': g1_res,
                    'labels': labels, 'is_bimodal': is_bimodal, 'g0_data': g0, 'g1_data': g1
                }
                progress_bar.progress((i + 1) / len(selected_filenames))

            progress_bar.empty()
            st.header("Global Summary Stats")

            if summary_data:
                full_df = pd.DataFrame(summary_data)
                default_cols = ["No.", "Filename", "Detected", "Stable Idx", "Duration (ms)",
                                "Abs Delta", "% Change", "T-Stat", "P-Value", "Modality"]
                all_cols = list(full_df.columns)
                selected_cols = st.multiselect("Select Columns to Display:", all_cols,
                                               default=[c for c in default_cols if c in all_cols])
                st.dataframe(full_df[selected_cols], use_container_width=True)

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for fname in selected_filenames:
                        if fname in analysis_results:
                            res = analysis_results[fname]
                            d_series = res['data']
                            g_info = res['global']
                            if g_info['has_warmup'] and 'end_idx' in g_info:
                                slice_idx = g_info['end_idx']
                                sliced_data = d_series.iloc[slice_idx:]
                            else:
                                sliced_data = d_series
                            csv_data = sliced_data.to_csv(
                                index=False, header=['value'])
                            zip_file.writestr(f"stable_{fname}", csv_data)
                st.download_button("📥 Download Stable Data (ZIP)", data=zip_buffer.getvalue(),
                                   file_name="stable_data_export.zip", mime="application/zip")

                # Loop through individual results
                for fname in selected_filenames:
                    if fname not in analysis_results:
                        continue
                    res = analysis_results[fname]
                    data = res['data']
                    is_bimodal = res['is_bimodal']
                    global_res = res['global']
                    g0_res, g1_res = res['g0'], res['g1']
                    labels = res['labels']
                    g0, g1 = res['g0_data'], res['g1_data']

                    st.divider()
                    st.header(f"Analysis: {fname}")

                    tab_list = ["Global Analysis", "Dist. Over Time"]
                    if is_bimodal:
                        tab_list.extend(
                            ["Global Rolling Means", "Group 0 (Slow)", "Group 1 (Fast)"])
                    tab_list.append("Global Raw Data")
                    tabs = st.tabs(tab_list)

                    with tabs[0]:
                        overlays = []
                        if is_bimodal:
                            if g0_res and g0_res['has_warmup']:
                                overlays.append(
                                    {'start_label': g0_res['start_label'], 'end_label': g0_res['end_label'],
                                     'color': 'tomato', 'label': 'Group 0'})
                            if g1_res and g1_res['has_warmup']:
                                overlays.append(
                                    {'start_label': g1_res['start_label'], 'end_label': g1_res['end_label'],
                                     'color': 'dodgerblue', 'label': 'Group 1'})
                        subgroups = {'g0': g0_res,
                                     'g1': g1_res} if is_bimodal else None
                        render_analysis_tab(data, global_res, "Global", 'black', show_gmm, show_iqr,
                                            group_labels=labels if is_bimodal else None, simple_view=is_bimodal,
                                            overlays=overlays, subgroup_res=subgroups)

                    with tabs[1]:
                        fig_ts, ax_ts = plt.subplots(figsize=(10, 8))
                        plot_time_slice_distributions(
                            ax_ts, data, window_size=utils.CONFIG['time_slice_window'])
                        st.pyplot(fig_ts)

                    curr_idx = 2
                    if is_bimodal:
                        with tabs[curr_idx]:
                            fig_roll, ax_roll = plt.subplots(figsize=(10, 5))
                            plot_global_rolling_means(ax_roll, data, g0, g1, global_res,
                                                      window_size=utils.CONFIG['rolling_mean_window'])
                            st.pyplot(fig_roll)
                        curr_idx += 1
                        with tabs[curr_idx]:
                            render_analysis_tab(g0, g0_res, "Group 0", 'darkred', show_gmm, show_iqr,
                                                group_labels=labels)
                        curr_idx += 1
                        with tabs[curr_idx]:
                            render_analysis_tab(g1, g1_res, "Group 1", 'navy', show_gmm, show_iqr,
                                                group_labels=labels)
                        curr_idx += 1

                    with tabs[curr_idx]:
                        starts, ends = [], []
                        if g0_res and g0_res['has_warmup']:
                            starts.append(g0_res['start_label'])
                            ends.append(g0_res['end_label'])
                        if g1_res and g1_res['has_warmup']:
                            starts.append(g1_res['start_label'])
                            ends.append(g1_res['end_label'])
                        widest_info = global_res.copy()
                        if starts and ends:
                            widest_info.update(
                                {'has_warmup': True, 'start_label': min(starts), 'end_label': max(ends),
                                 'warm_dur': max(ends) - min(starts)})
                        fig, ax = plt.subplots(figsize=(12, 4))
                        plot_regime(ax, data, widest_info, "Global Raw Dataset (Widest Warmup Highlight)", 'gray',
                                    group_labels=None, simple_view=False, show_trend=False, show_iqr=False)
                        st.pyplot(fig)
