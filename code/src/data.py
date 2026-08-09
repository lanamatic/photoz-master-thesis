"""Loading and preparation of the GalaxiesML dataset.

The official split shipped with the dataset is roughly 70/15/15
(train/validation/test) and contains no calibration set. Since conformal
prediction requires one, the official validation set is halved here: one half
stays validation, the other becomes calibration. Train and test are left
untouched, keeping results directly comparable to Jones et al. (2024).

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

FILES = {
    "train": "5x127x127_training_with_morphology.csv",
    "val": "5x127x127_validation_with_morphology.csv",
    "test": "5x127x127_testing_with_morphology.csv",
}

# --- columns ----------------------------------------------------------------
BANDS = ["g", "r", "i", "z", "y"]

TARGET = "specz_redshift"
TARGET_ERR = "specz_redshift_err"
FLAG = "specz_flag_homogeneous"
OBJ_ID = "object_id"

MAG_COLS = [f"{b}_cmodel_mag" for b in BANDS]
MAGERR_COLS = [f"{b}_cmodel_magsigma" for b in BANDS]

# adjacent filter pairs -> colours
COLOR_PAIRS = [("g", "r"), ("r", "i"), ("i", "z"), ("z", "y")]
COLOR_COLS = [f"{a}_{b}" for a, b in COLOR_PAIRS]

# baseline feature set: 5 magnitudes + 4 colours
FEATURES = MAG_COLS + COLOR_COLS

# morphological features (kept for later experiments, unused in the baseline)
MORPH_KINDS = [
    "sersic_index",
    "half_light_radius",
    "ellipticity",
    "isophotal_area",
    "petro_rad",
    "peak_surface_brightness",
]
MORPH_COLS = [f"{b}_{k}" for b in BANDS for k in MORPH_KINDS]

# redshift ranges used for binned evaluation
Z_BINS = [0.0, 0.5, 1.0, 2.0, np.inf]
Z_BIN_LABELS = ["0-0.5", "0.5-1", "1-2", ">2"]

SEED = 42


# =============================================================================
#  Loading
# =============================================================================

def load_raw(split: str) -> pd.DataFrame:
    """Load one official split, keeping only the columns we need."""
    path = DATA_DIR / FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    keep = {OBJ_ID, TARGET, TARGET_ERR, FLAG, *MAG_COLS, *MAGERR_COLS, *MORPH_COLS}
    return pd.read_csv(path, usecols=lambda c: c in keep)


def apply_quality_cuts(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Quality cuts following Jones et al. (2024), Table 1.

    The dataset ships pre-cleaned, so this normally removes nothing - but bad
    rows slipping through silently would corrupt the metrics, so we check.
    """
    n0 = len(df)
    mask = (
        df[TARGET].notna()
        & (df[TARGET] > 0)
        & (df[TARGET] < 9.99)                      # 9.9999 flags "unknown"
        & df[TARGET_ERR].between(0, 1, inclusive="neither")
        & df[FLAG].astype(bool)
        & df[MAG_COLS].notna().all(axis=1)
        & (df[MAG_COLS] > 0).all(axis=1)           # negative magnitudes = bad
        & (df[MAG_COLS] < 50).all(axis=1)          # sentinel values such as 99
    )
    out = df.loc[mask].reset_index(drop=True)

    if verbose and len(out) < n0:
        print(f"  quality cuts: dropped {n0 - len(out):,} of {n0:,} rows")
    return out


def add_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add colours as magnitude differences between adjacent filters."""
    df = df.copy()
    for a, b in COLOR_PAIRS:
        df[f"{a}_{b}"] = df[f"{a}_cmodel_mag"] - df[f"{b}_cmodel_mag"]
    return df


def add_z_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Add the redshift-range label used for binned evaluation."""
    df = df.copy()
    df["z_bin"] = pd.cut(df[TARGET], bins=Z_BINS, labels=Z_BIN_LABELS, right=False)
    return df


def load_split(split: str, verbose: bool = True) -> pd.DataFrame:
    """Load one split, apply quality cuts, add colours and redshift bins."""
    if verbose:
        print(f"Loading {split} ...")
    df = load_raw(split)
    df = apply_quality_cuts(df, verbose=verbose)
    df = add_colors(df)
    return add_z_bin(df)


# =============================================================================
#  Split: validation -> validation + calibration
# =============================================================================

def split_val_calib(
    df_val: pd.DataFrame, calib_frac: float = 0.5, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the official validation set into validation and calibration.

    The split is random, not stratified by redshift. Conformal prediction
    assumes exchangeability between the calibration and test sets; stratifying
    would change the calibration distribution relative to test and break the
    very assumption the coverage guarantee rests on.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df_val))
    n_cal = int(round(calib_frac * len(df_val)))

    calib = df_val.iloc[idx[:n_cal]].reset_index(drop=True)
    val = df_val.iloc[idx[n_cal:]].reset_index(drop=True)
    return val, calib


def load_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Return all four splits: train, val, calib, test."""
    train = load_split("train", verbose)
    val_full = load_split("val", verbose)
    test = load_split("test", verbose)
    val, calib = split_val_calib(val_full)
    return {"train": train, "val": val, "calib": calib, "test": test}


# =============================================================================
#  Standardization
# =============================================================================

@dataclass
class Standardizer:
    """Feature standardisation, backed by sklearn's StandardScaler.

    Mean and standard deviation are computed on the training set only -
    anything else leaks information from the test set. Only the fitted
    mean_/scale_ arrays are persisted (not the sklearn object itself), so
    saved scalers stay loadable even across sklearn version upgrades.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        scaler = StandardScaler.fit(X)
        self.mean = scaler.mean_
        self.std = scaler.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Standardizer is not fitted (call .fit first)")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "Standardizer":
        d = np.load(path)
        return cls(mean=d["mean"], std=d["std"])


def to_arrays(
    df: pd.DataFrame, features: list[str] = FEATURES
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dataframe to (X, y) float32 arrays."""
    X = df[features].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X, y


# =============================================================================
#  Summary
# =============================================================================

def summarize(splits: dict[str, pd.DataFrame]) -> None:
    total = sum(len(d) for d in splits.values())

    print("\n" + "=" * 72)
    print("SPLIT SIZES")
    print("=" * 72)
    for name, d in splits.items():
        print(f"  {name:<8} {len(d):>8,}  ({100 * len(d) / total:5.2f} %)")
    print(f"  {'total':<8} {total:>8,}")

    print("\n" + "=" * 72)
    print("REDSHIFT")
    print("=" * 72)
    for name, d in splits.items():
        z = d[TARGET]
        print(f"  {name:<8} min={z.min():6.3f}  median={z.median():6.3f}  "
              f"max={z.max():6.3f}  mean={z.mean():6.3f}")

    print("\n" + "=" * 72)
    print("DISTRIBUTION ACROSS REDSHIFT RANGES")
    print("=" * 72)
    header = f"  {'range':<8}" + "".join(f"{n:>16}" for n in splits)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label in Z_BIN_LABELS:
        row = f"  {label:<8}"
        for d in splits.values():
            n = int((d["z_bin"] == label).sum())
            row += f"{n:>9,} ({100 * n / len(d):4.1f}%)"
        print(row)

    print("\n" + "=" * 72)
    print("FEATURES (training set)")
    print("=" * 72)
    tr = splits["train"]
    for c in FEATURES:
        s = tr[c]
        print(f"  {c:<16} mean={s.mean():8.4f}  std={s.std():7.4f}  "
              f"[{s.min():7.3f}, {s.max():7.3f}]")


def check_calibration_bins(
    splits: dict[str, pd.DataFrame], width: float = 0.1, zmax: float = 4.0
) -> None:
    """Report how many calibration galaxies fall into each fixed-width bin.

    Too few galaxies in a bin makes the quantile estimate noisy, and below
    ceil(1/alpha) - 1 galaxies the quantile index does not exist at all, so no
    valid interval can be constructed at that confidence level.
    """
    cal = splits["calib"]
    edges = np.arange(0, zmax + width, width)
    counts = pd.cut(cal[TARGET], bins=edges).value_counts().sort_index()

    print("\n" + "=" * 72)
    print(f"CALIBRATION BINS  (width {width}, {len(counts)} bins)")
    print("=" * 72)
    print(f"  galaxies total:      {len(cal):,}")
    print(f"  mean per bin:        {len(cal) / len(counts):.0f}")
    print(f"  empty bins:          {(counts == 0).sum()}")
    print(f"  bins with < 30:      {(counts < 30).sum()}")
    print(f"  bins with < 100:     {(counts < 100).sum()}")

    # smallest bin size still admitting a valid quantile, per coverage level
    print("\n  Feasibility by coverage level:")
    for alpha, level in [(0.317, "68.3%"), (0.10, "90%"), (0.05, "95%")]:
        n_min = int(np.ceil(1 / alpha)) - 1
        n_bad = int((counts < n_min).sum())
        print(f"    {level:>6}  needs n >= {n_min:>3}  ->  {n_bad} bin(s) infeasible")

    weak = counts[counts < 100]
    if len(weak):
        print("\n  Bins with fewer than 100 galaxies:")
        for interval, n in weak.items():
            print(f"    {interval.left:4.1f}-{interval.right:4.1f}  {n:>5,}")

    # alternative: equal-count bins
    for target_n in (300, 500):
        k = max(int(len(cal) / target_n), 1)
        q = np.quantile(cal[TARGET], np.linspace(0, 1, k + 1))
        print(f"\n  Quantile bins targeting ~{target_n} galaxies each: {k} bins")
        print(f"    narrowest: {np.diff(q).min():.4f}   "
              f"widest: {np.diff(q).max():.4f}   "
              f"edges: {q[0]:.2f} ... {q[-1]:.2f}")


if __name__ == "__main__":
    data = load_all()
    summarize(data)
    check_calibration_bins(data)