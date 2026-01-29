import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy import stats
from typing import Tuple, Dict
import utils
import warmup # For detect_warmup

## COMPARISON STATISTICAL UTILITIES

def run_statistical_suite(a: pd.Series, b: pd.Series, alpha: float = 0.05) -> Tuple[Dict, float, float]:
    """
    Runs 7 robust tests and returns detailed statistics + WEIGHTED score + MAX possible score.
    Saves NULL DISTRIBUTION for Permutation Test.
    """
    results = {}
    total_score = 0.0
    max_possible_score = sum(utils.CONFIG['weights'].values())

    # Helper to format results and apply weights
    def add_res(name, p, stat, extra="", null_dist=None):
        nonlocal total_score

        # Handle errors/nans
        if np.isnan(p) or np.isnan(stat):
            results[name] = {"P-Value": 1.0, "Statistic": 0.0, "Verdict": "Error", "Details": "NaN Result", "Weight": 0}
            return

        is_diff = p < alpha
        weight = utils.CONFIG['weights'].get(name, 1.0)

        if is_diff:
            total_score += weight

        res_dict = {
            "P-Value": p,
            "Statistic": stat,
            "Verdict": "Different" if is_diff else "Same",
            "Details": extra,
            "Weight": weight
        }

        # Capture Null Distribution for plotting
        if null_dist is not None:
            res_dict['null_dist'] = null_dist

        results[name] = res_dict

    # 1) Mann-Whitney U
    try:
        res = stats.mannwhitneyu(a, b, alternative='two-sided')
        add_res('Mann-Whitney U', res.pvalue, res.statistic, f"u={res.statistic:.1f}")
    except:
        results['Mann-Whitney U'] = {"Verdict": "Error", "P-Value": 1.0, "Statistic": 0.0}

    # 2) Kolmogorov-Smirnov
    try:
        res = stats.ks_2samp(a, b)
        add_res('Kolmogorov-Smirnov', res.pvalue, res.statistic, f"D={res.statistic:.4f}")
    except:
        results['Kolmogorov-Smirnov'] = {"Verdict": "Error", "P-Value": 1.0, "Statistic": 0.0}

    # 3) Cramer-von Mises
    try:
        res = stats.cramervonmises_2samp(a, b)
        add_res('Cramer-von Mises', res.pvalue, res.statistic, f"T={res.statistic:.4f}")
    except:
        results['Cramer-von Mises'] = {"Verdict": "Error", "P-Value": 1.0, "Statistic": 0.0}

    # 4) Anderson-Darling
    try:
        res = stats.anderson_ksamp([a, b])
        add_res('Anderson-Darling', res.significance_level, res.statistic, f"A^2={res.statistic:.4f}")
    except:
        results['Anderson-Darling'] = {"Verdict": "Error", "P-Value": 1.0, "Statistic": 0.0}

    # 5) Epps-Singleton
    try:
        res = stats.epps_singleton_2samp(a, b)
        add_res('Epps-Singleton', res.pvalue, res.statistic, f"W2={res.statistic:.2f}")
    except:
        results['Epps-Singleton'] = {"Verdict": "Error", "P-Value": 1.0, "Statistic": 0.0}

    # 6) Permutation Test
    try:
        res = stats.permutation_test((a, b), utils.diff_means_statistic, n_resamples=utils.CONFIG['perm_resamples'],
                                     alternative='two-sided')
        add_res('Permutation', res.pvalue, res.statistic, f"Diff={res.statistic:.4f}", null_dist=res.null_distribution)
    except:
        results['Permutation'] = {"Verdict": "Error", "P-Value": 1.0, "Statistic": 0.0}

    # 7) Brunner-Munzel
    try:
        res = stats.brunnermunzel(a, b)
        add_res('Brunner-Munzel', res.pvalue, res.statistic, f"W={res.statistic:.2f}")
    except:
        results['Brunner-Munzel'] = {"Verdict": "Error", "P-Value": 1.0, "Statistic": 0.0}

    return results, total_score, max_possible_score


def analyze_chunk_pair(chunk_a, chunk_b, tolerance):
    """
    Compares two chunks using WEIGHTED statistical tests + Hodges-Lehmann Logic.
    """
    multi_a, means_a = utils.check_modality(chunk_a)
    multi_b, means_b = utils.check_modality(chunk_b)

    clean_a = utils.remove_outliers_robust(chunk_a)
    clean_b = utils.remove_outliers_robust(chunk_b)

    # Statistical Tests with Weighting
    test_results, total_score, max_score = run_statistical_suite(clean_a, clean_b, utils.CONFIG['alpha'])

    med_a = clean_a.median()
    med_b = clean_b.median()
    if med_a == 0: med_a = 1e-9

    hl_shift = utils.get_hodges_lehmann_delta(clean_a.values, clean_b.values)
    gain_pct = (hl_shift / med_a) * 100
    abs_gain = abs(gain_pct)

    # Logic: Score > 50% of max possible score
    threshold_score = max_score / 2
    is_statistically_diff = total_score > threshold_score

    winner = "Tie"
    reason = "Statistically Similar"

    gain_str = f"Diff: {abs_gain:.2f}% (HL)"

    if is_statistically_diff:
        if abs_gain < tolerance:
            winner = "Tie"
            reason = f"Diff ({abs_gain:.2f}%) < Tol. | Score: {total_score}/{max_score}"
        elif med_b < med_a:
            winner = "B"
            reason = f"B Lower Median (B is {abs_gain:.2f}% faster) | Score: {total_score}/{max_score}"
        else:
            winner = "A"
            reason = f"A Lower Median (B is {abs_gain:.2f}% slower) | Score: {total_score}/{max_score}"
    else:
        winner = "Tie"
        reason = f"Distributions Similar | Score: {total_score}/{max_score}"

    return winner, reason, (multi_a, means_a), (multi_b, means_b), gain_pct, clean_a, clean_b, test_results


## VISUALIZATION FUNCTIONS (COMPARISON SPECIFIC)

def plot_chunk_ribbon(ax, results_log, total_len):
    """
    Plots a horizontal bar chart summarizing chunk winners over time.
    """
    ax.set_xlim(0, total_len)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Iteration")
    ax.set_title("Timeline of Statistical Winners (Chunk Analysis)", fontweight='bold')

    for res in results_log:
        start, end, w = res['start'], res['end'], res['winner']
        if w == 'A':
            color = 'tomato'
        elif w == 'B':
            color = 'dodgerblue'
        else:
            color = 'lightgray'

        ax.broken_barh([(start, end - start)], (0, 1), facecolors=color, edgecolor='white')
        mid = (start + end) / 2
        ax.text(mid, 0.5, w, ha='center', va='center', color='white' if w != 'Tie' else 'black', fontweight='bold',
                fontsize=8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='tomato', label='API A Wins'),
        mpatches.Patch(facecolor='dodgerblue', label='API B Wins'),
        mpatches.Patch(facecolor='lightgray', label='Tie')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def plot_ecdf(ax, data_a, data_b):
    """Plots Empirical Cumulative Distribution Function for forensic analysis."""
    # Sort data
    x_a = np.sort(data_a)
    y_a = np.arange(1, len(x_a) + 1) / len(x_a)
    x_b = np.sort(data_b)
    y_b = np.arange(1, len(x_b) + 1) / len(x_b)

    # Plot
    ax.step(x_a, y_a, label='API A', where='post', color='red', alpha=0.8, linewidth=2)
    ax.step(x_b, y_b, label='API B', where='post', color='blue', alpha=0.8, linewidth=2)

    # Visual Polish
    ax.set_title("ECDF (Cumulative Probability)", fontweight='bold')
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Prob <= X")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_box_comparison(ax, data_a, data_b):
    """Plots Notched Box Plot for median comparison."""
    data = [data_a, data_b]

    ax.boxplot(data, notch=True, patch_artist=True,
               boxprops=dict(facecolor='lightgray', color='black'),
               medianprops=dict(color='red', linewidth=2),
               labels=['API A', 'API B'])
    ax.set_title("Distribution Spread (Notched Box Plot)", fontweight='bold')
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, axis='y', alpha=0.3)


def plot_permutation_visual(ax, observed_diff, null_dist):
    """
    Histogram of Permutation Test Null Distribution vs Observed Diff.
    """

    ax.hist(null_dist, bins=30, color='lightgray', edgecolor='white', density=True, label='Null Dist')
    ax.axvline(observed_diff, color='red', linewidth=2.5, linestyle='--', label=f'Observed: {observed_diff:.2f}')

    # Calculate p-value visually for label (approx)
    p_val_approx = np.mean(np.abs(null_dist) >= np.abs(observed_diff))

    ax.set_title(f"Permutation Test (Diff of Means)\np ~ {p_val_approx:.3f}", fontweight='bold')
    ax.set_xlabel("Difference in Means")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_violin_comparison(ax, data_a, data_b):
    """
    Violin plot specifically for Brunner-Munzel (Variance check).
    """

    parts = ax.violinplot([data_a, data_b], showmeans=False, showmedians=True)

    # Style
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    parts['bodies'][0].set_facecolor('tomato')  # A
    parts['bodies'][1].set_facecolor('dodgerblue')  # B
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['API A', 'API B'])
    ax.set_title("Variance Shape (Violin Plot for Brunner-Munzel)", fontweight='bold')
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, axis='y', alpha=0.3)


## MAIN TAB FUNCTIONALITY (COMPARISON)

def render_comparison_tab(chunk_size, perf_tolerance):
    st.header("A/B Benchmark Comparison Protocol")
    st.markdown(
        "Compare two API runs. Uses the **Code 1 Warmup Detection** logic to identify steady state start, then performs chunk analysis using **7 robust non-parametric tests**.")

    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader("Upload Baseline (API A)", type=['csv'], key="tab2_a")
    with col_b:
        file_b = st.file_uploader("Upload Comparison (API B)", type=['csv'], key="tab2_b")

    if file_a and file_b:
        try:
            # Load Data
            file_a.seek(0)
            file_b.seek(0)
            df_a = pd.read_csv(file_a).iloc[:, 0].dropna().reset_index(drop=True)
            df_b = pd.read_csv(file_b).iloc[:, 0].dropna().reset_index(drop=True)

            # 1. Warmup detection (using warmup.py)
            with st.status("Running Analysis...", expanded=True) as status:
                st.write("Detecting Regime Changes (Using Code 1 Method)...")

                # Call warmup.detect_warmup
                res_a = warmup.detect_warmup(df_a, "auto")
                res_b = warmup.detect_warmup(df_b, "auto")

                # Extract end indices
                idx_a = res_a['end_idx'] if res_a and res_a['has_warmup'] else 0
                idx_b = res_b['end_idx'] if res_b and res_b['has_warmup'] else 0

                stats_a = utils.calculate_warmup_stats(df_a, idx_a)
                stats_b = utils.calculate_warmup_stats(df_b, idx_b)

                st.write("Slicing Steady State data...")
                steady_a = df_a.iloc[idx_a:].reset_index(drop=True)
                steady_b = df_b.iloc[idx_b:].reset_index(drop=True)

                # 2. Chunk Analysis
                st.write("Running Multi-Test Statistical Analysis per Chunk...")
                min_len = min(len(steady_a), len(steady_b))
                if min_len < chunk_size:
                    st.warning("Data too short for defined chunk size.")
                    num_chunks = 0
                else:
                    num_chunks = min_len // chunk_size

                results_log = []
                votes = {'A': 0, 'B': 0, 'Tie': 0}

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = start + chunk_size
                    winner, reason, info_a, info_b, gain_pct, clean_a, clean_b, test_stats = analyze_chunk_pair(
                        steady_a.iloc[start:end],
                        steady_b.iloc[start:end],
                        tolerance=perf_tolerance
                    )
                    votes[winner] += 1
                    results_log.append({
                        'chunk': i + 1, 'start': start, 'end': end,
                        'winner': winner, 'reason': reason, 'gain_pct': gain_pct,
                        'info_a': info_a, 'info_b': info_b,
                        'clean_a': clean_a, 'clean_b': clean_b,
                        'test_stats': test_stats
                    })

                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # SCORECARD
            final_winner = "Tie"

            # Check for clear performance winner
            if votes['A'] > votes['B'] and votes['A'] > votes['Tie']:
                final_winner = "API A (Better Steady State Performance)"
            elif votes['B'] > votes['A'] and votes['B'] > votes['Tie']:
                final_winner = "API B (Better Steady State Performance)"
            else:
                # Performance is tied/similar. Check warmup duration.
                if idx_b < idx_a:
                    final_winner = "API B (Faster Stabilization)"
                elif idx_a < idx_b:
                    final_winner = "API A (Faster Stabilization)"
                else:
                    final_winner = "Tie (Identical Performance & Stabilization)"

            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Winner", final_winner)
            col2.metric("Warm-up: API A", f"{idx_a} cycles")
            col3.metric("Warm-up: API B", f"{idx_b} cycles", delta=idx_a - idx_b, delta_color="normal")

            # Show Vote Tally
            st.info(f"Chunk Votes: API A ({votes['A']}) | API B ({votes['B']}) | Tie ({votes['Tie']})")

            # Ribbon chart (for overall view)
            st.subheader("Timeline of Analysis")
            fig_ribbon, ax_ribbon = plt.subplots(figsize=(10, 1.5))
            plot_chunk_ribbon(ax_ribbon, results_log, min_len)
            st.pyplot(fig_ribbon)

            # Per-each-chunk view
            st.markdown("---")
            st.subheader("Visual Diagnostics (Chunk by Chunk View)")

            # Create dropdown for chunk selection
            chunk_options = [f"Chunk {r['chunk']} ({r['winner']})" for r in results_log]
            selected_chunk_str = st.selectbox("Select Chunk to Analyze", chunk_options)

            if selected_chunk_str:
                # Parse selection
                chunk_idx = int(selected_chunk_str.split(" ")[1]) - 1
                res = results_log[chunk_idx]

                st.markdown(f"#### Analysis for Chunk {res['chunk']}")
                st.write(f"**Winner:** {res['winner']}")
                st.write(f"**Reason:** {res['reason']}")

                # Detailed evidence table
                st.markdown("##### Statistical Evidence")

                # Convert nested dict to DataFrame
                # We dropped 'null_dist' from the table display because it's an array
                table_data = {k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k != 'null_dist'}
                              for k, v in res['test_stats'].items()}

                evidence_df = pd.DataFrame(table_data).T

                # Format for display (color code the Verdict)
                def style_verdict(v):
                    color = 'red' if v == 'Different' else 'green'
                    return f'color: {color}; font-weight: bold'

                # Streamlit's newer column config
                try:
                    st.dataframe(
                        evidence_df.style.map(style_verdict, subset=['Verdict'])
                        .format("{:.4f}", subset=['P-Value', 'Statistic'])
                    )
                except:
                    st.table(evidence_df)

                # Detailed graphs for the selected chunk
                c_graph1, c_graph2 = st.columns(2)

                with c_graph1:
                    fig_ecdf, ax_ecdf = plt.subplots(figsize=(6, 4))
                    plot_ecdf(ax_ecdf, res['clean_a'], res['clean_b'])
                    st.pyplot(fig_ecdf)

                with c_graph2:
                    fig_box, ax_box = plt.subplots(figsize=(6, 4))
                    plot_box_comparison(ax_box, res['clean_a'], res['clean_b'])
                    st.pyplot(fig_box)

                # Permutation histogram & Violin plot
                c_graph3, c_graph4 = st.columns(2)
                with c_graph3:
                    # Extract data for Permutation plot
                    perm_res = res['test_stats'].get('Permutation', {})
                    null_dist = perm_res.get('null_dist')
                    obs_stat = perm_res.get('Statistic')

                    fig_perm, ax_perm = plt.subplots(figsize=(6, 4))
                    if null_dist is not None:
                        plot_permutation_visual(ax_perm, obs_stat, null_dist)
                    else:
                        ax_perm.text(0.5, 0.5, "Permutation Data N/A", ha='center')
                    st.pyplot(fig_perm)

                with c_graph4:
                    fig_vio, ax_vio = plt.subplots(figsize=(6, 4))
                    plot_violin_comparison(ax_vio, res['clean_a'], res['clean_b'])
                    st.pyplot(fig_vio)

            st.markdown("---")

            # Visualization (for overall scatter plots)
            st.subheader("Visual Diagnostics (Global View)")
            fig, axes = plt.subplots(4, 1, figsize=(12, 24))
            plt.subplots_adjust(hspace=0.3)

            # Plot 1: API-A Warmup (Using Code 1 Visualization style logic inside Code 2 structure)
            axes[0].scatter(df_a.index, df_a, color='red', s=2, alpha=0.3, label='Raw Data')
            axes[0].axvline(idx_a, c='black', ls='--', lw=2, label=f'End: {idx_a}')
            axes[0].axvspan(0, idx_a, color='red', alpha=0.1)
            axes[0].set_title(f"1. Baseline (API A) Warm-up", fontweight='bold')
            axes[0].set_ylabel("Latency")
            axes[0].legend(loc='upper right')

            # Plot 2: API-B Warmup
            axes[1].scatter(df_b.index, df_b, color='blue', s=2, alpha=0.3, label='Raw Data')
            axes[1].axvline(idx_b, c='black', ls='--', lw=2, label=f'End: {idx_b}')
            axes[1].axvspan(0, idx_b, color='blue', alpha=0.1)
            axes[1].set_title(f"2. Comparison (API B) Warm-up", fontweight='bold')
            axes[1].set_ylabel("Latency")
            axes[1].legend(loc='upper right')

            # PLOT 3: Showing the modality of distribution with relevant medians
            axes[2].scatter(steady_a[:min_len].index, steady_a[:min_len], color='red', s=2, alpha=0.2,
                            label='A (Raw)')
            axes[2].scatter(steady_b[:min_len].index, steady_b[:min_len], color='blue', s=2, alpha=0.2,
                            label='B (Raw)')

            for res in results_log:
                s, e, w = res['start'], res['end'], res['winner']
                c_a, c_b = res['clean_a'], res['clean_b']
                multi_a, modes_a = res['info_a']
                multi_b, modes_b = res['info_b']
                bg_color = {'A': 'red', 'B': 'blue', 'Tie': 'gray'}[w]
                axes[2].axvspan(s, e, color=bg_color, alpha=0.1)
                axes[2].scatter(c_a.index, c_a, color='darkred', s=5, alpha=0.45)
                axes[2].scatter(c_b.index, c_b, color='navy', s=5, alpha=0.45)
                # Visual Only: Multimodality lines
                for m in modes_a: axes[2].hlines(m, s, e, colors='#ec0d7c', linestyles='--')
                for m in modes_b: axes[2].hlines(m, s, e, colors='cyan', linestyles='--')
                mid = (s + e) / 2
                axes[2].text(mid, axes[2].get_ylim()[1], w, ha='center', va='top', alpha=0.5, fontweight='bold')

            axes[2].set_title(f"3. Steady State Analysis (Cleaned Points Highlighted)", fontweight='bold')
            custom_lines = [
                Line2D([0], [0], color='darkred', marker='o', linestyle='', label='A Clean'),
                Line2D([0], [0], color='navy', marker='o', linestyle='', label='B Clean')
            ]
            axes[2].legend(handles=custom_lines, loc='upper right')

            # Plot 4: Distribution graphs for API-A and API-B
            sns.kdeplot(steady_a[:min_len], ax=axes[3], color='red', fill=True, alpha=0.1, label='API A')
            sns.kdeplot(steady_b[:min_len], ax=axes[3], color='blue', fill=True, alpha=0.1, label='API B')
            med_a = steady_a.median()
            med_b = steady_b.median()
            overall_gain = ((med_a - med_b) / med_a) * 100 if med_a != 0 else 0
            status_txt = "Faster" if overall_gain > 0 else "Slower"
            axes[3].set_title(f"4. Overall Latency Distribution (B is {abs(overall_gain):.2f}% {status_txt})",
                              fontweight='bold')
            axes[3].legend()

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during analysis: {e}")