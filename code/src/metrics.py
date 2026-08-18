"""Point and probabilistic metrics for photometric redshift evaluation.

All formulas follow Jones et al. (2024a), Table 2, to keep results
comparable.

"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from src.data import Z_BIN_LABELS, Z_BINS
    
GAMMA = 0.15                    # bandwidth of the robust loss
OUTLIER_THRESHOLD = 0.15        # |dz| / (1 + z_spec) > this -> outlier
CATASTROPHIC_THRESHOLD = 1.0    # |dz| > this -> catastrophic outlier
TARGET_COVERAGE = 0.683         # 1-sigma level used throughout this thesis


# =============================================================================
#  Point metrics
# =============================================================================

def _delta_z(z_phot: np.ndarray, z_spec: np.ndarray) -> np.ndarray:
    return z_phot - z_spec

def _normalized_dz(z_phot: np.ndarray, z_spec: np.ndarray) -> np.ndarray:
    return _delta_z(z_phot, z_spec) / (1.0 + z_spec)

def outlier_fraction(z_phot: np.ndarray, z_spec: np.ndarray) -> float:
    return float(np.mean(np.abs(_normalized_dz(z_phot, z_spec)) > OUTLIER_THRESHOLD))
 
def catastrophic_fraction(z_phot: np.ndarray, z_spec: np.ndarray) -> float:
    return float(np.mean(np.abs(_delta_z(z_phot, z_spec)) > CATASTROPHIC_THRESHOLD))
 
def rms(z_phot: np.ndarray, z_spec: np.ndarray) -> float:
    return float(np.sqrt(np.mean(_normalized_dz(z_phot, z_spec) ** 2)))
 
def bias(z_phot: np.ndarray, z_spec: np.ndarray) -> float:
    return float(np.mean(_normalized_dz(z_phot, z_spec)))

def scatter(z_phot: np.ndarray, z_spec: np.ndarray) -> float:
    """
    Robust scatter: sigma_MAD = 1.4826 * MAD of dz.
    Defined on raw dz, not normalized dz, matching Jones et al. (2024a).
    """
    dz = _delta_z(z_phot, z_spec)
    mad = np.median(np.abs(dz - np.median(dz)))
    return float(1.4826 * mad)

def point_loss(z_phot: np.ndarray, z_spec: np.ndarray, gamma: float = GAMMA) -> float:
    """Tanaka-style loss: mean of 1 - 1 / (1 + (dz / gamma)^2)."""
    dz = _delta_z(z_phot, z_spec)
    return float(np.mean(1.0 - 1.0 / (1.0 + (dz / gamma) ** 2)))

def point_metrics(z_phot: np.ndarray, z_spec: np.ndarray) -> dict[str, float]:
    """All point metrics as a dict."""
    return {
        "N": int(len(z_phot)),
        "outlier": outlier_fraction(z_phot, z_spec),
        "catastrophic": catastrophic_fraction(z_phot, z_spec),
        "rms": rms(z_phot, z_spec),
        "bias": bias(z_phot, z_spec),
        "scatter": scatter(z_phot, z_spec),
        "loss": point_loss(z_phot, z_spec),
    }


# =============================================================================
#  Interval-based probabilistic metrics
# =============================================================================

def outlier_3sigma_fraction(
    z_phot: np.ndarray, z_spec: np.ndarray, sigma: np.ndarray
) -> float:
    return float(np.mean(np.abs(_delta_z(z_phot, z_spec)) > 3.0 * sigma))

def coverage(
    z_spec: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """
    Fraction of true values falling inside [lower, upper].

    Target for a 1-sigma interval is 0.683. Above -> intervals too wide
    (uncertainty overestimated); below -> too narrow (underestimated).
    """
    return float(np.mean((z_spec >= lower) & (z_spec <= upper)))

def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))
 
def gaussian_interval(
    z_phot: np.ndarray, z_spec: np.ndarray, sigma: np.ndarray, k: float = 1.0
) -> float:
    """Coverage of the +/- k*sigma interval (Gaussian intervals)."""
    return coverage(z_spec, z_phot - k * sigma, z_phot + k * sigma)

def pit_values(
    z_phot: np.ndarray, z_spec: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """
    Probability Integral Transform, Gaussian case.

    PIT_i = CDF_i(z_spec_i) = Phi((z_spec - mu) / sigma).
    
    A well-calibrated model gives a flat PIT histogram;
    A central peak -> PDFs too broad; 
    Peaks at the edges -> too narrow or many catastrophic outliers.
    """
    from scipy.stats import norm
    return norm.cdf((z_spec - z_phot) / sigma)

def pit_values_from_cdf(z_spec: np.ndarray, cdf_at_spec: np.ndarray) -> np.ndarray:
    """
    PIT when the predictive CDF evaluated at z_spec is available directly 
    (general (non-Gaussian) predictive CDF). 
 
    Needed for MDN, whose predictive distribution is a Gaussian mixture and is
    not summarised by a single sigma.
    """
    return np.asarray(cdf_at_spec, dtype=float)

def crps_gaussian(
    z_phot: np.ndarray, z_spec: np.ndarray, sigma: np.ndarray
) -> float:
    """
    Mean CRPS under a Gaussian predictive distribution (closed form).
 
    CRPS(N(mu, sigma), y) = sigma * [ w*(2*Phi(w) - 1) + 2*phi(w) - 1/sqrt(pi) ],
    with w = (y - mu) / sigma. Lower is better (Gneiting & Raftery 2007)
    """
    from scipy.stats import norm
    sigma = np.asarray(sigma, dtype=float)
    w = (z_spec - z_phot) / sigma
    crps = sigma * (w * (2 * norm.cdf(w) - 1) + 2 * norm.pdf(w) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))

def crps_ensemble(z_spec: np.ndarray, samples: np.ndarray) -> float:
    """Sample-based CRPS, for any predictive distribution given draws from it.
 
    `samples` has shape (n_galaxies, n_samples). 
    Uses the energy form CRPS = E|X - y| - 0.5 * E|X - X'|, estimated per galaxy. 
    (Gneiting & Raftery 2007)

    Works for MDN, MC-Dropout and BNN predictive samples.
    """
    z_spec = np.asarray(z_spec, dtype=float)
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[1]
 
    term1 = np.mean(np.abs(samples - z_spec[:, None]), axis=1)
    s = np.sort(samples, axis=1)
    weights = (2 * np.arange(1, n + 1) - n - 1)
    term2 = (2.0 / (n * n)) * np.sum(weights[None, :] * s, axis=1)
 
    return float(np.mean(term1 - 0.5 * term2))
 


# =============================================================================
#  Evaluation per redshift range(bin)
#
#  Every metric is computed both over the whole test set and separately per
#  redshift range (data.Z_BIN_LABELS).
# =============================================================================

def assign_bins(z_spec: np.ndarray) -> pd.Series:
    """Assign each galaxy to a redshift range."""
    return pd.Series(pd.cut(z_spec, bins=Z_BINS, labels=Z_BIN_LABELS, right=False))
 
def metrics_by_bin(
    z_phot: np.ndarray,
    z_spec: np.ndarray,
    sigma: np.ndarray | None = None,
    include_overall: bool = True,
) -> pd.DataFrame:
    """
    Compute every applicable metric per redshift range and overall.
 
    Point metrics are always computed. 
    Probabilistic ones (3-sigma fraction, 1-sigma coverage) are added only when `sigma` is given. 
    
    Returns a DataFrame indexed by range label,
    with an "overall" row when requested.
    """
    z_phot = np.asarray(z_phot, dtype=float)
    z_spec = np.asarray(z_spec, dtype=float)
    bins = assign_bins(z_spec)
 
    def row(mask: np.ndarray) -> dict[str, float]:
        d = point_metrics(z_phot[mask], z_spec[mask])
        if sigma is not None:
            sig = np.asarray(sigma, dtype=float)[mask]
            d["cov_1sigma"] = gaussian_interval(
                z_phot[mask], z_spec[mask], sig, k=1.0
            )
            d["width_1sigma"] = mean_interval_width(
                z_phot[mask] - sig, z_phot[mask] + sig
            )
            d["frac_3sigma"] = outlier_3sigma_fraction(
                z_phot[mask], z_spec[mask], sig
            )
        return d
 
    rows: dict[str, dict[str, float]] = {}
    for label in Z_BIN_LABELS:
        mask = (bins == label).to_numpy()
        if mask.any():
            rows[label] = row(mask)
    if include_overall:
        rows["overall"] = row(np.ones(len(z_spec), dtype=bool))
 
    return pd.DataFrame(rows).T

# =============================================================================
#  LaTeX export
# =============================================================================

def to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    cols: list[str] | None = None,
    float_fmt: str = "{:.4f}",
) -> str:
    """
    Render a metrics DataFrame as a LaTeX table.
 
    Write the result into code/results/*.tex and \\input it from a chapter.
    """
    if cols is not None:
        df = df[cols]
 
    def fmt(col: str, v) -> str:
        if col == "N":
            return f"{int(v):,}".replace(",", ".")
        try:
            return float_fmt.format(float(v)).replace(".", ",")
        except (ValueError, TypeError):
            return str(v)

    header = " & ".join(["Распон"] + list(df.columns)) + r" \\"
    body = [
        " & ".join([str(idx)] + [fmt(col, v) for col, v in zip(df.columns, r)]) + r" \\"
        for idx, r in df.iterrows()
    ]
 
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "r" * len(df.columns) + "}",
        r"\toprule", header, r"\midrule",
        *body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])
 