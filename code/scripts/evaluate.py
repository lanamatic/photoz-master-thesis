"""Evaluate a trained model on the test set.
 
Loads a run saved by train.py, computes point (and, where available,
probabilistic) metrics overall and per redshift range, and writes a LaTeX
table into results/. 

For the NN baseline the key check is RMS ~0.17 and near-zero bias - matching Jones et al. (2024a).
 
Run from code/:
    python -m scripts.evaluate --run nn_seed42
"""
 
from __future__ import annotations
 
import argparse
import json
from pathlib import Path
 
import numpy as np
import torch
 
from src.data import FEATURES, ROOT, Standardizer, load_all, to_arrays
from src.metrics import metrics_by_bin, to_latex
from src.models import build_model
 
RESULTS_DIR = ROOT / "code" / "results"
 
# Jones et al. (2024a), Table 3, photometry-only NN - overall test set.
JONES_NN = {"outlier": 0.059, "catastrophic": 0.029, "rms": 0.174,
            "bias": 0.0001, "scatter": 0.026, "loss": 0.089}
 
 
def load_run(run_name: str):
    """Restore model + standardizer + config from a saved run."""
    d = RESULTS_DIR / run_name
    cfg = json.loads((d / "config.json").read_text())
    scaler = Standardizer.load(d / "scaler.npz")
 
    model = build_model(cfg["model"], in_dim=len(cfg["features"]),
                        dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    model.eval()
    return model, scaler, cfg
 
 
def evaluate(run_name: str) -> None:
    model, scaler, cfg = load_run(run_name)
 
    splits = load_all(verbose=False)
    X_te, y_te = to_arrays(splits["test"], FEATURES)
    X_te = scaler.transform(X_te)
    x = torch.from_numpy(X_te)
 
    is_mc = cfg["model"] == "mc_dropout"
    if is_mc:
        mean, std, _ = model.predict_mc(x, n_samples=100)
        z_pred = mean.numpy()
        sigma = std.numpy()
    else:
        z_pred = model.predict(x).numpy()
        sigma = None
 
    table = metrics_by_bin(z_pred, y_te, sigma)

    print(f"\nRun: {run_name}   ({cfg['model']}, best epoch {cfg['best_epoch']})")
    print("=" * 72)
    print(table.round(4).to_string())
 
    # Comparison to Jones for the baseline
    if cfg["model"] in ("nn", "mc_dropout"):
        ov = table.loc["overall"]
        print("\nOverall vs Jones et al. (2024a) NN:")
        print(f"  {'metric':<14}{'ours':>10}{'Jones':>10}{'diff':>10}")
        for k, jones_v in JONES_NN.items():
            ours = float(ov[k])
            print(f"  {k:<14}{ours:>10.4f}{jones_v:>10.4f}{ours - jones_v:>+10.4f}")
 
    # Export LaTeX
    out = RESULTS_DIR / f"metrics_{run_name}.tex"
    cols = ["N", "outlier", "catastrophic", "rms", "bias", "scatter"]
    if sigma is not None:
        cols += ["cov_1sigma", "width_1sigma", "frac_3sigma"]
    latex = to_latex(
        table, cols=cols,
        caption=f"Метрике на тест скупу, модел {cfg['model']}.",
        label=f"tab:metrics-{cfg['model']}",
    )
    out.write_text(latex, encoding="utf-8")
    print(f"\nLaTeX table -> {out}")
 
 
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run name, e.g. nn_seed42")
    evaluate(p.parse_args().run)
 
 
if __name__ == "__main__":
    main()