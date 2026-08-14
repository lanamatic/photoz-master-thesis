"""Exploratory analysis of the GalaxiesML dataset.

Produces the figures for the "Подаци" chapter and writes them straight into
thesis/figures/, so the thesis always compiles against current plots without
manual copying.

"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from src.data import (
    COLOR_COLS,
    FEATURES,
    MAG_COLS,
    ROOT,
    TARGET,
    Z_BIN_LABELS,
    load_all,
)

FIG_DIR = ROOT / "thesis" / "figures"
RESULTS_DIR = ROOT / "code" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# consistent styling across every figure in the thesis
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})

PALETTE = ["#1f4e79", "#65a958", "#c98b3a", "#873838"]


def fig_nz(splits) -> None:
    """Redshift distribution N(z) - the key figure of the data chapter."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(0, 4, 120)

    ax = axes[0]
    for name, color in zip(["train", "val", "calib", "test"], PALETTE):
        ax.hist(splits[name][TARGET], bins=bins, histtype="step",
                lw=1.4, color=color, label=name, density=True)
    ax.set_xlabel(r"$z_\mathrm{spec}$")
    ax.set_ylabel("густина")
    ax.set_title("Расподела по скуповима")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.hist(splits["train"][TARGET], bins=bins, color=PALETTE[0], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel(r"$z_\mathrm{spec}$")
    ax.set_ylabel("број галаксија")
    ax.set_title("Тренинг скуп, логаритмаска скала")
    for edge in (0.5, 1.0, 2.0):          # evaluation range boundaries
        ax.axvline(edge, color="k", ls="--", lw=0.8, alpha=0.5)

    fig.savefig(FIG_DIR / "nz_distribucija.pdf")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'nz_distribucija.pdf'}")


def fig_colors_vs_z(splits) -> None:
    """Colours against redshift - illustrates the colour degeneracy."""
    tr = splits["train"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True)

    for ax, col in zip(axes, COLOR_COLS):
        ax.hexbin(tr[TARGET], tr[col], gridsize=70, bins="log",
                  cmap="viridis", mincnt=1, extent=(0, 4, -1.5, 3))
        ax.set_xlabel(r"$z_\mathrm{spec}$")
        ax.set_title(col.replace("_", " $-$ "))
    axes[0].set_ylabel("боја [mag]")

    fig.savefig(FIG_DIR / "boje_vs_z.pdf")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'boje_vs_z.pdf'}")


def fig_magnitudes(splits) -> None:
    """Magnitude distributions per filter."""
    tr = splits["train"]
    fig, ax = plt.subplots(figsize=(6, 3.4))

    for col in MAG_COLS:
        ax.hist(tr[col], bins=np.linspace(14, 30, 120), histtype="step",
                lw=1.3, label=col.split("_")[0], density=True)
    ax.set_xlabel("магнитуда")
    ax.set_ylabel("густина")
    ax.legend(frameon=False, ncol=5, fontsize=8)

    fig.savefig(FIG_DIR / "magnitude.pdf")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'magnitude.pdf'}")


def table_bins(splits) -> None:
    """LaTeX table of galaxy counts per redshift range (data chapter)."""
    order = ["train", "val", "calib", "test"]

    def fmt(n: int) -> str:                # Serbian thousands separator
        return f"{n:,}".replace(",", ".")

    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Распон & Тренинг & Валидација & Калибрација & Тест \\", r"\midrule",
    ]
    for label in Z_BIN_LABELS:
        cells = [fmt(int((splits[s]["z_bin"] == label).sum())) for s in order]
        lines.append(f"${label}$ & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        "Укупно & " + " & ".join(fmt(len(splits[s])) for s in order) + r" \\",
        r"\bottomrule", r"\end{tabular}",
    ]

    path = RESULTS_DIR / "tabela_binovi.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


def report_outliers(splits, n_sigma: float = 10.0) -> None:
    """Count extreme feature values - reported in the data chapter."""
    tr = splits["train"]
    print(f"\nExtreme values beyond {n_sigma:.0f} sigma:")
    total = 0
    for c in FEATURES:
        s = tr[c]
        n = int(((s - s.mean()).abs() > n_sigma * s.std()).sum())
        total += n
        if n:
            print(f"  {c:<16} {n:>5} rows")
    n_faint = int((tr[MAG_COLS] > 27).any(axis=1).sum())
    print(f"  {'magnitude > 27':<16} {n_faint:>5} rows")
    print(f"  total flagged: {total} of {len(tr):,} "
          f"({100 * total / len(tr):.3f} %)")


def report_correlations(splits) -> None:
    """Spearman correlation of each feature with redshift."""
    tr = splits["train"]
    print("\nSpearman correlation with redshift:")
    corr = tr[FEATURES + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
    for name, v in corr.sort_values(key=abs, ascending=False).items():
        print(f"  {name:<16} {v:+.3f}")


if __name__ == "__main__":
    data = load_all()

    print("\nGenerating figures:")
    fig_nz(data)
    fig_colors_vs_z(data)
    fig_magnitudes(data)
    table_bins(data)

    report_outliers(data)
    report_correlations(data)