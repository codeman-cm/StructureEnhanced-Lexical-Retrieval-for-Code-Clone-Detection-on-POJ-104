from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "tfidf_token": "Token",
    "tfidf_token_normid": "Norm-Token",
    "ast_proxy_only": "Structure",
    "gac_tfidf_structure": "Token+Struct",
    "gac_tfidf_structure_normid": "Norm+Struct",
}


def setup_style():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["Times New Roman", "Arial"]
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["xtick.major.width"] = 0.9
    plt.rcParams["ytick.major.width"] = 0.9
    plt.rcParams["axes.grid"] = False
    plt.rcParams["legend.frameon"] = True


def finish_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
    ax.minorticks_off()
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def save_figure(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def gradient_colors(values, cmap_name="coolwarm"):
    values = np.asarray(values, dtype=float)
    norm = mpl.colors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = mpl.colormaps[cmap_name]
    return [cmap(norm(v)) for v in values]


def plot_dataset_statistics(stats_df: pd.DataFrame, out_dir: Path):
    setup_style()
    df = stats_df.copy()
    splits = df["split"].tolist()
    x = np.arange(len(splits))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    bar_vals = df["rows"].to_numpy()
    colors = ["#174A7C", "#B54A4A", "#2F6B4F"]
    bars = ax1.bar(x - 0.17, bar_vals / 1000.0, width=0.34, color=colors,
                    edgecolor="black", linewidth=0.8, label="Samples")
    ax1.set_ylabel("Samples (x10^3)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in splits], fontsize=11)
    ax1.set_ylim(0, max(bar_vals / 1000.0) * 1.22)
    for rect, value in zip(bars, bar_vals):
        ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.6,
                 f"{int(value):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2 = ax1.twinx()
    line1, = ax2.plot(x + 0.17, df["labels"], color="#7A1E1E", marker="s",
                      linewidth=1.6, markersize=5, label="Labels")
    line2, = ax2.plot(x + 0.17, df["mean_code_tokens"], color="#0E3A5B", marker="o",
                      linewidth=1.6, markersize=5, linestyle="--", label="Mean tokens")
    ax2.set_ylabel("Labels / Mean tokens", fontsize=12)
    ax2.set_ylim(0, max(df["mean_code_tokens"]) * 1.18)
    finish_axes(ax1)
    finish_axes(ax2)
    leg = ax1.legend([bars, line1, line2], ["Samples", "Labels", "Mean tokens"],
                     loc="upper right", framealpha=0.92, edgecolor="black", fontsize=9)
    leg.get_frame().set_linewidth(0.6)
    save_figure(fig, out_dir, "fig3_dataset_statistics")


def plot_main_comparison(summary: dict, out_dir: Path):
    setup_style()
    models = ["tfidf_token", "tfidf_token_normid", "ast_proxy_only", "gac_tfidf_structure_normid"]
    metrics = ["MAP@R", "Recall@10", "MRR"]
    labels = [MODEL_LABELS[m] for m in models]
    palette = ["#173B63", "#9A2E2E", "#2E6F57", "#111111"]
    markers = ["o", "s", "^", "D"]
    y = np.arange(len(metrics))
    offsets = np.linspace(-0.18, 0.18, len(models))
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for i, model in enumerate(models):
        vals = np.array([summary["models"][model][metric]["mean"] for metric in metrics])
        ypos = y + offsets[i]
        ax.plot(vals, ypos, linestyle="none", marker=markers[i], markersize=7,
                markeredgecolor="black", markeredgewidth=0.6,
                color=palette[i], label=labels[i], zorder=3)
        for value, yy in zip(vals, ypos):
            ax.hlines(yy, 0.15, value, color=palette[i], linewidth=1.0, alpha=0.24, zorder=1)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.set_xlim(0.15, 1.03)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.55, len(metrics) - 0.45)
    finish_axes(ax)
    leg = ax.legend(loc="lower right", framealpha=0.92, edgecolor="black", fontsize=9, ncol=2)
    leg.get_frame().set_linewidth(0.6)
    save_figure(fig, out_dir, "fig4_main_metric_comparison")


def plot_ablation_mapr(summary: dict, out_dir: Path):
    setup_style()
    models = ["tfidf_token", "tfidf_token_normid", "ast_proxy_only",
              "gac_tfidf_structure", "gac_tfidf_structure_normid"]
    labels = [MODEL_LABELS[m] for m in models]
    means = np.array([summary["models"][m]["MAP@R"]["mean"] for m in models])
    stds = np.array([summary["models"][m]["MAP@R"]["std"] for m in models])
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(x, means, yerr=stds, fmt="-o", color="#173B63", ecolor="#111111",
                elinewidth=0.9, capsize=3, markersize=6.5,
                markerfacecolor="#C0002B", markeredgecolor="black", markeredgewidth=0.7,
                linewidth=1.4, zorder=3)
    ax.fill_between(x, means - stds, means + stds, color="#173B63", alpha=0.10, zorder=1)
    for xx, value in zip(x, means):
        ax.text(xx, value + 0.012, f"{value:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("MAP@R", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=10)
    ax.set_ylim(0.16, 0.50)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5])
    ax.set_xlim(-0.35, len(models) - 0.65)
    finish_axes(ax)
    save_figure(fig, out_dir, "fig5_ablation_mapr")


def plot_seed_stability(per_seed_metrics: pd.DataFrame, out_dir: Path):
    setup_style()
    df = per_seed_metrics.copy()
    df = df[df["model"].isin(["tfidf_token_normid", "gac_tfidf_structure_normid"])].copy()
    df["seed"] = df["seed"].astype(str)
    seeds = ["42", "2026", "3407"]
    base_vals = np.array([float(df[(df["model"] == "tfidf_token_normid") & (df["seed"] == s)]["MAP@R"].iloc[0]) for s in seeds])
    main_vals = np.array([float(df[(df["model"] == "gac_tfidf_structure_normid") & (df["seed"] == s)]["MAP@R"].iloc[0]) for s in seeds])
    gains = main_vals - base_vals
    mean_gain = float(gains.mean())
    std_gain = float(gains.std(ddof=0))
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.axhspan(mean_gain - std_gain, mean_gain + std_gain, color="#173B63", alpha=0.10, zorder=0)
    ax.axhline(mean_gain, color="#173B63", linewidth=1.3, linestyle="--", zorder=1, label="Mean gain")
    ax.plot(x, gains, linestyle="none", marker="o", markersize=7,
            markerfacecolor="#C0002B", markeredgecolor="black", markeredgewidth=0.8,
            color="#C0002B", zorder=3, label="Seed gain")
    for xx, value in zip(x, gains):
        ax.text(xx, value + 0.00010, f"{value:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("MAP@R gain", fontsize=12)
    ax.set_xlabel("Random seed", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=10)
    ax.set_xlim(-0.45, len(seeds) - 0.55)
    span = max(0.0006, std_gain * 4 + 0.0004)
    ax.set_ylim(mean_gain - span, mean_gain + span)
    finish_axes(ax)
    leg = ax.legend(loc="lower right", framealpha=0.92, edgecolor="black", fontsize=9)
    leg.get_frame().set_linewidth(0.6)
    save_figure(fig, out_dir, "fig6_seed_stability")


def plot_structure_gain(summary: dict, out_dir: Path):
    setup_style()
    base = summary["models"]["tfidf_token_normid"]
    main = summary["models"]["gac_tfidf_structure_normid"]
    metrics = ["MAP@R", "Recall@1", "Recall@5", "Recall@10", "MRR"]
    gains = np.array([main[m]["mean"] - base[m]["mean"] for m in metrics])
    y = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = gradient_colors(gains, "coolwarm")
    for yy, value, color in zip(y, gains, colors):
        ax.hlines(yy, 0.0, value, color=color, linewidth=2.0, alpha=0.82, zorder=2)
        ax.plot(value, yy, marker="o", markersize=8, color=color,
                markeredgecolor="black", markeredgewidth=0.8, zorder=3)
        ax.text(value + 0.004, yy, f"{value:.3f}", ha="left", va="center",
                fontsize=9, fontweight="bold")
    ax.axvline(0.0, color="#111111", linewidth=0.9)
    ax.set_xlabel("Absolute score gain", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=10)
    ax.set_xlim(0.0, max(gains) * 1.26)
    ax.set_xticks([0.0, 0.03, 0.06, 0.09, 0.12])
    ax.set_ylim(-0.55, len(metrics) - 0.45)
    finish_axes(ax)
    save_figure(fig, out_dir, "fig7_structure_gain")
