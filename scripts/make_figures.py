"""Figures of the paper: one column wide (8.2 cm), all text 10 pt Times New Roman.

Usage (from the repository root):
    python scripts/make_figures.py

Writes paper/figures/Fig<k>_<name>.png (600 dpi). Palette: dataviz reference slots blue,
orange, aqua (validated all-pairs on white); every series also has its own marker and a legend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_paper_tables import FAMILY, R, results  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, WASH = "#0b0b0b", "#3a3936", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
WC = 3.23     # inches = 8.2 cm, one journal column
FS = 10       # points, journal rule for all text in figures

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix", "font.size": FS,
    "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 600, "figure.facecolor": "white", "axes.facecolor": "white", "axes.titleweight": "normal"})


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y" if grid == "x" else "x", visible=False)
    ax.tick_params(length=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"{name}.png")


def fig1_protocol():
    fig, ax = plt.subplots(figsize=(WC, 3.9))
    ax.set_xlim(0, 100), ax.set_ylim(0, 100), ax.axis("off")

    def box(x, y, w, h, text, accent=False, bold_first=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2", linewidth=0.8,
                                    edgecolor=BLUE if accent else AXIS, facecolor="#eaf2fc" if accent else WASH))
        lines = text.split("\n")
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=FS, color=INK,
                linespacing=1.15)

    def arrow(x0, y0, x1, y1, color=INK2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=color, lw=0.8))

    box(18, 88, 64, 10, "UNSW-NB15 official partitions")
    box(1, 66, 30, 14, "Training\npart (80%)")
    box(35, 66, 30, 14, "Validation\npart (20%)")
    box(69, 66, 30, 14, "Test\npartition", accent=True)
    arrow(40, 87.4, 16, 80.6), arrow(50, 87.4, 50, 80.6), arrow(60, 87.4, 84, 80.6)
    box(1, 36, 64, 22, "Search and model choice\n(10 seeds, equal budgets);\nshift analysis; diagnostics")
    arrow(16, 65.4, 16, 58.6), arrow(50, 65.4, 50, 58.6)
    arrow(76, 65.4, 60, 58.6)
    ax.text(76.5, 52, "features", ha="center", va="center", fontsize=FS, color=INK2)
    box(18, 4, 64, 22, "Final evaluation, once:\ntest macro-F1, hypotheses,\nbinary view, split effect", accent=True)
    arrow(33, 35.4, 40, 26.6)
    ax.plot([88, 88], [65.4, 15], color=BLUE, lw=0.8)
    arrow(88, 15, 82.6, 15, color=BLUE)
    ax.text(89.5, 40, "labels", rotation=90, ha="left", va="center", fontsize=FS, color=BLUE)
    save(fig, "Fig1_protocol")


def fig2_val_test():
    rows = []
    for r in results("e2_unsw_baselines"):
        rows.append((r["model"], r["val"]["macro_f1"], r["test"]["macro_f1"]))
    for r in results("e1b_unsw_gbdt"):
        rows.append(("hgb_default" if r["tuner"] == "default" else f"hgb_{r['tuner']}", r["val"]["macro_f1"],
                     r["test"]["macro_f1"]))
    for r in results("e1_unsw_tuners"):
        rows.append((f"qrnn_{r['tuner']}", r["val"]["macro_f1"], r["test"]["macro_f1"]))
    df = pd.DataFrame(rows, columns=["model", "val", "test"]).groupby("model").mean()
    fam = {"tree": (BLUE, "o", "Trees and boosting"), "neural": (ORANGE, "s", "MLP, CNN-BiLSTM"),
           "qrnn": (AQUA, "^", "QRNN"), "linear": (MUTED, "D", "Logistic regression")}

    def f(m):
        x = FAMILY.get(m, "")
        return "qrnn" if "QRNN" in x else "neural" if x == "neural" else "linear" if x == "linear" else "tree"

    fig, ax = plt.subplots(figsize=(WC, 3.0))
    lim = (0.28, 0.68)
    ax.plot(lim, lim, color=MUTED, lw=0.8)
    ax.text(0.405, 0.418, "test = validation", rotation=45, transform_rotates_text=True, rotation_mode="anchor",
            fontsize=FS, color=INK2, ha="center", va="bottom")
    ax.plot(lim, (lim[0] - 0.11, lim[1] - 0.11), color=AXIS, lw=0.8)
    ax.text(0.505, 0.385, "0.11 lower", rotation=45, transform_rotates_text=True, rotation_mode="anchor",
            fontsize=FS, color=INK2, ha="center", va="top")
    for key, (c, mk, lab) in fam.items():
        d = df[[f(m) == key for m in df.index]]
        ax.scatter(d.val, d.test, s=26, color=c, marker=mk, edgecolors="white", linewidths=0.6, label=lab, zorder=3)
    ax.set_xlim(*lim), ax.set_ylim(0.28, 0.58)
    ax.set_xlabel("Validation macro-F1")
    ax.set_ylabel("Test macro-F1")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.1)
    save(fig, "Fig2_val_test")


def fig3_shift():
    gr = pd.read_csv(R / "diag_shift" / "greedy.csv")
    auc = pd.read_csv(R / "diag_shift" / "domain_auc.csv").groupby("comparison")["auc"].mean()
    target = 0.5 + (gr.auc.iloc[0] - 0.5) / 2
    names = ["none removed"] + [str(d) for d in gr.dropped.dropna()]
    names = names[:len(gr)]
    y = np.arange(len(gr))[::-1]
    fig, ax = plt.subplots(figsize=(WC, 3.1))
    ax.axvline(auc["control"], color=MUTED, lw=0.8)
    ax.axvline(target, color=MUTED, lw=0.8)
    ax.plot(gr.auc, y, color=BLUE, lw=1.4, marker="o", ms=3.5, mec="white", mew=0.6, zorder=3)
    ax.set_yticks(y, names)
    ax.set_xlim(0.45, 0.95)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Domain AUC (class mix matched)")
    ax.text(auc["control"] + 0.006, len(gr) - 0.6, "no shift", fontsize=FS, color=INK2, ha="left")
    ax.text(target + 0.006, len(gr) - 0.6, "half", fontsize=FS, color=INK2, ha="left")
    ax.set_ylim(-0.7, len(gr) + 0.1)
    clean(ax, grid="y")
    ax.grid(axis="x", visible=True)
    save(fig, "Fig3_shift")


def fig4_e6():
    short = {"default": "Default", "std_hsboa": "HSBOA, standard", "iw_hsboa": "HSBOA, shift",
             "iw_sboa": "SBOA, shift", "iw_pso": "PSO, shift", "iw_random": "RS, shift",
             "iwtrain_hsboa": "HSBOA, shift + w.t."}
    df = pd.DataFrame([{"arm": r["arm"], "val": r["val"]["macro_f1"], "test": r["test"]["macro_f1"]}
                       for r in results("e6_unsw_shift")])
    order = df.groupby("arm")["test"].mean().sort_values().index
    fig, axes = plt.subplots(2, 1, figsize=(WC, 4.4), sharey=True)
    for ax, col, title, xl in zip(axes, ("val", "test"), ("(a) Validation", "(b) Test partition"),
                                  ((0.59, 0.665), (0.49, 0.548))):
        for i, a in enumerate(order):
            c, mk = (BLUE, "o") if a == "default" else (ORANGE, "s")
            s = df[df.arm == a][col].to_numpy()
            ax.scatter(s, np.full(len(s), i), s=7, color=c, alpha=0.35, marker=mk, linewidths=0)
            ax.scatter(s.mean(), i, s=30, color=c, marker=mk, edgecolors="white", linewidths=0.7, zorder=3)
        ax.set_yticks(range(len(order)), [short[a] for a in order])
        ax.set_xlim(*xl)
        ax.set_title(title, loc="left", fontsize=FS)
        clean(ax)
    axes[1].set_xlabel("Macro-F1 (large mark = mean of 10 seeds)")
    fig.subplots_adjust(hspace=0.45)
    save(fig, "Fig4_e6")


def fig5_split():
    p = pd.read_csv(R / "final_eval" / "protocol.csv")
    p = p[p.settings == "default"]
    fig, axes = plt.subplots(1, 2, figsize=(WC, 2.4))
    for ax, col, title in zip(axes, ("macro_f1", "binary_accuracy"), ("(a) Macro-F1", "(b) Binary accuracy")):
        wide = p.pivot(index="seed", columns="split", values=col)
        for _, r in wide.iterrows():
            ax.plot([0, 1], [r["official"], r["random_70_30"]], color=AXIS, lw=0.8, zorder=1)
        m = wide.mean()
        ax.plot([0, 1], [m["official"], m["random_70_30"]], color=BLUE, lw=1.4, marker="o", ms=4, mec="white",
                mew=0.6, zorder=3)
        ax.set_xticks([0, 1], ["Official", "Random"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(title, loc="left", fontsize=FS)
        clean(ax, grid="y")
        ax.grid(axis="y", visible=True)
    fig.subplots_adjust(wspace=0.6)
    save(fig, "Fig5_split")


if __name__ == "__main__":
    fig1_protocol()
    fig2_val_test()
    fig3_shift()
    fig4_e6()
    fig5_split()
