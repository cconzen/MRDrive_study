import pandas as pd
import numpy as np
from itertools import combinations

import Path
import re
import seaborn as sns
import matplotlib.pyplot as plt

from statannotations.Annotator import Annotator
from scipy import stats

from scipy.stats import ttest_rel

from statsmodels.stats.multitest import multipletests


plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")

CONDITIONS   = ["Baseline", "Proactive", "Optional"]
DVS          = ["trust", "understanding", "nasa"]
DV_LABELS    = {"trust": "Trust", "understanding": "System Understanding", "nasa": "Mental Workload"}
BLOCKS       = [1, 2, 3]
ALPHA_BONF   = 0.05 / 3
SUBJECT_COL  = "participant_id"

# IBM palette preselection for the conditions
COLORS = {
    "Lumo": "#6929c4",
    "Coda": "#1192e8",
    "Nelo": "#005d5d",
    "Optional": "#6929c4",
    "Proactive": "#1192e8",
    "Baseline": "#005d5d",
}

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def paired_ttests(df, dv, subject_col="participant_id"):
    wide = (
        df.pivot_table(
            index=subject_col,
            columns="condition",
            values=dv,
            aggfunc="mean"
        )
        .dropna(subset=CONDITIONS)
    )

    pairs = list(combinations(CONDITIONS, 2))

    t_values = []
    p_values = []

    for c1, c2 in pairs:
        t, p = stats.ttest_rel(wide[c1], wide[c2])
        t_values.append(t)
        p_values.append(p)

    _, p_adj, _, _ = multipletests(p_values, method="holm") # can change to bonferroni

    results = {}

    for (c1, c2), t, p_raw, p_holm in zip(pairs, t_values, p_values, p_adj):
        results[(c1, c2)] = {
            "t": t,
            "p_raw": p_raw,
            "p_holm": p_holm,
            "sig": sig_stars(p_holm),
            "n": len(wide),
        }

    return results

def plot_qq_by_condition(df, dv, condition_col="condition", title=None):

    conditions = df[condition_col].unique()

    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5))

    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        subset = df[df[condition_col] == cond][dv].dropna()

        stats.probplot(subset, dist="norm", plot=ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ccc")
        ax.spines["bottom"].set_color("#ccc")
        ax.tick_params(axis='x', which='both', length=4, width=1, color='black')
        ax.tick_params(axis='y', which='both', length=4, width=1, color='black',        direction='out', labelsize=10 )
        ax.set_facecolor("#FFFFFF")
        ax.get_lines()[0].set_markerfacecolor("#BFD7EA")
        ax.get_lines()[0].set_markeredgecolor("#087E8B")
        ax.grid(axis="y", color="#ddd", linewidth=0.5, linestyle="--")
        ax.grid(axis="x", color="#ddd", linewidth=0.5, linestyle="--")
        ax.get_lines()[1].set_color("#C81D25")

        ax.set_title(f"{title} – {cond}")

    plt.tight_layout()
    plt.show()

def stylised_boxplot(df, metric):
    """
    Parameters
    ----------
    df : pd.DataFrame
    metric : str
        One of {"trust", "understanding", "nasa"}.
    """

    # can change axis labels here! brackets are drawn for the pairs defined; pre-selected sig pairs only
    settings = {
        "trust": {
            "ylabel": "Trust Score",
            "pairs": [("Optional", "Baseline")], # sig
            "print_stats": False,
        },
        "understanding": {
            "ylabel": "System Understanding Score",
            "pairs": [
                ("Optional", "Baseline"), # sig
                ("Proactive", "Baseline")
            ],
            "print_stats": False,
        },
        "nasa": {
            "ylabel": "Mental Workload (NASA-TLX)",
            "pairs": [], # none are sig, so no brackets
            "print_stats": True,
        },
    }

    if metric not in settings:
        raise ValueError(f"metric must be one of {list(settings.keys())}")

    cfg = settings[metric]

    rng = np.random.default_rng(42)
    plt.rcParams["font.sans-serif"] = ["Roboto"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

    # Boxplot
    sns.boxplot(
        data=df,
        x="condition",
        y=metric,
        hue="condition",
        order=CONDITIONS,
        hue_order=CONDITIONS,
        palette=COLORS,
        width=0.6,
        fliersize=0,
        linewidth=1,
        boxprops=dict(alpha=0.45),
        medianprops=dict(color="#111", linewidth=1),
        whiskerprops=dict(color="#666", linewidth=1),
        capprops=dict(color="#666", linewidth=1),
        ax=ax,
    )


    for i, c in enumerate(CONDITIONS):
        vals = df[df["condition"] == c][metric].dropna().values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))

        mean_val = vals.mean()
        # this adds the scatterplots to the boxplots
        ax.scatter(
            np.full_like(vals, i) + jitter,
            vals,
            color=COLORS[c],
            s=20,
            alpha=0.5,
            zorder=4,
            linewidths=0,
        )
        # this adds the mean dashed lines
        ax.hlines(
            mean_val,
            i - 0.3,
            i + 0.3,
            colors=COLORS[c], # in the colour of the condition
            linestyles="--",
            linewidth=1,
            zorder=5,
        )

    # to check / print stats
    if cfg["print_stats"]:
        summary = (
            df.groupby("condition")[metric]
            .agg(["mean", "std", "count"])
            .reindex(CONDITIONS)
        )

        print(f"\n{metric.upper()} descriptive statistics:")
        for condition, row in summary.iterrows():
            print(
                f"{condition}: "
                f"M = {row['mean']:.2f}, "
                f"SD = {row['std']:.2f}, "
                f"N = {int(row['count'])}"
            )


    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, fontsize=12, color="black")
    ax.set_xlabel("")
    ax.set_ylabel(cfg["ylabel"], fontsize=12)


    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", color="#ddd", linewidth=0.5, linestyle="--")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")

    # loop to change the box outlines to their condition colours
    for patch, c in zip(ax.patches, CONDITIONS):
        patch.set_edgecolor(COLORS[c])
        patch.set_linewidth(1.2)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # *** bracket annotations
    if cfg["pairs"]:
        annotator = Annotator(
            ax,
            cfg["pairs"],
            data=df,
            x="condition",
            y=metric,
        )

        annotator.configure(
            test="t-test_paired",
            comparisons_correction="bonferroni", # for some reason 'holm' uses the uncorrected p values
            text_format="star", # "single"
            loc="outside",
        )

        annotator.apply_and_annotate()

    ax.tick_params(
        axis="x",
        which="both",
        length=4,
        width=1,
        color="black",
    )

    ax.tick_params(
        axis="y",
        which="both",
        length=4,
        width=1,
        color="black",
        direction="out",
        labelsize=10,
    )

    plt.xlabel("Explanation Condition", fontsize=14)
    plt.ylabel(cfg["ylabel"], fontsize=14)

    plt.tight_layout()
    plt.show()

#print(st.stats.ComparisonsCorrection.methods_names.keys())

def stylised_boxplot_grouped(df, metric, group_colors=None):

    GROUP_ORDER = ["No Explanation", "Explanation"]
    if group_colors is None:
        group_colors = {"No Explanation": "#6929c4", "Explanation": "#1192e8"}

    settings = {
        "trust": {"ylabel": "Trust Score", "print_stats": True},
        "understanding": {"ylabel": "System Understanding", "print_stats": True},
        "nasa": {"ylabel": "Mental Workload (NASA-TLX)", "print_stats": True},
    }
    if metric not in settings:
        raise ValueError(f"metric must be one of {list(settings.keys())}")
    cfg = settings[metric]

    df_grouped = df.copy()
    df_grouped["explanation_group"] = df_grouped["condition"].replace({
        "Optional": "Explanation",
        "Proactive": "Explanation",
        "Baseline": "No Explanation",
    })

    summary = (df_grouped.groupby(["participant_id", "explanation_group"])[metric].mean().reset_index())

    rng = np.random.default_rng(42)
    plt.rcParams["font.sans-serif"] = ["Roboto"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)

    sns.boxplot(
        data=summary,
        x="explanation_group",
        y=metric,
        hue="explanation_group",
        order=GROUP_ORDER,
        palette=group_colors,
        width=0.5,
        fliersize=0,
        linewidth=1,
        boxprops=dict(alpha=0.45),
        medianprops=dict(color="#111", linewidth=1),
        whiskerprops=dict(color="#666", linewidth=1),
        capprops=dict(color="#666", linewidth=1),
        ax=ax,
        legend=False,
    )

    # scatter + mean dashed lines
    for i, g in enumerate(GROUP_ORDER):
        vals = summary[summary["explanation_group"] == g][metric].dropna().values
        jitter = rng.uniform(-0.1, 0.1, size=len(vals))
        mean_val = vals.mean()

        ax.scatter(
            np.full_like(vals, i) + jitter,
            vals,
            color=group_colors[g],
            s=24,
            alpha=0.5,
            zorder=4,
            linewidths=0,
        )
        ax.hlines(
            mean_val,
            i - 0.25,
            i + 0.25,
            colors=group_colors[g],
            linestyles="--",
            linewidth=1,
            zorder=5,
        )

    if cfg["print_stats"]:
        desc = (
            summary.groupby("explanation_group")[metric]
            .agg(["mean", "std", "count"])
            .reindex(GROUP_ORDER)
        )
        print(f"\n{metric.upper()} descriptive statistics (grouped):")
        for group, row in desc.iterrows():
            print(f"{group}: M = {row['mean']:.2f}, SD = {row['std']:.2f}, N = {int(row['count'])}")

        pivot = summary.pivot(index="participant_id", columns="explanation_group", values=metric)
        t_stat, p_val = ttest_rel(pivot["Explanation"], pivot["No Explanation"])
        print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER, fontsize=10, color="black")
    ax.set_xlabel("")
    ax.set_ylabel(cfg["ylabel"], fontsize=12)

    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", color="#ddd", linewidth=0.5, linestyle="--")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")

    for patch, g in zip(ax.patches, GROUP_ORDER[::-1]): # idk why i have to reverse this
        patch.set_edgecolor(group_colors[g])
        patch.set_linewidth(1.2)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # significance bracket for the single pairwise comparison
    annotator = Annotator(
        ax,
        [("No Explanation", "Explanation")],
        data=summary,
        x="explanation_group",
        y=metric,
        order=GROUP_ORDER,
    )
    annotator.configure(
        test="t-test_paired",
        text_format="star",
        loc="outside",
    )
    annotator.apply_and_annotate()

    ax.tick_params(axis="x", which="both", length=4, width=1, color="black")
    ax.tick_params(axis="y", which="both", length=4, width=1, color="black",direction="out", labelsize=10,)

    plt.xlabel("Explanation Condition", fontsize=12)
    plt.ylabel(cfg["ylabel"], fontsize=12)

    plt.tight_layout()
    plt.show()


def print_json_structure():

    DATA_FOLDER = "trimmed_data"

    VIDEO_PATTERN = re.compile(r"varjo_capture_(.+?)_trimmed\.mp4$")
    GAZE_PATTERN = re.compile(r"varjo_gaze_output_(.+?)_trimmed\.csv$")

    participants = []

    root = Path(DATA_FOLDER)

    for participant_dir in sorted(root.iterdir(), key=lambda p: int(p.name)):
        if not participant_dir.is_dir():
            continue

        participant_id = f"P{int(participant_dir.name):02d}"

        video_files = sorted(participant_dir.glob("varjo_capture_*_trimmed.mp4"))

        videos = []

        for block_idx, video_file in enumerate(video_files):
            match = VIDEO_PATTERN.match(video_file.name)

            if not match:
                continue

            timestamp = match.group(1)

            gaze_file = participant_dir / f"varjo_gaze_output_{timestamp}_trimmed.csv"

            if not gaze_file.exists():
                print(f"Missing gaze file for: {video_file.name}")
                continue

            videos.append({
                "block": str(block_idx),
                "video": f'{{DATA_FOLDER}}/{participant_dir.name}/{video_file.name}',
                "gaze": f'{{DATA_FOLDER}}/{participant_dir.name}/{gaze_file.name}',
            })

        participants.append({
            "id": participant_id,
            "videos": videos
        })

    print("PARTICIPANTS = [")

    for participant in participants:
        print("    {")
        print(f'        "id": "{participant["id"]}",')
        print('        "videos": [')

        for video in participant["videos"]:
            print(
                f'            {{"block": "{video["block"]}", '
                f'"video": f"{video["video"]}", '
                f'"gaze": f"{video["gaze"]}"}},'
            )

        print("        ]")
        print("    },")

    print("]")