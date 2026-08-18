"""
Training loop for the photometric redshift models.
 
Reproducing the Jones et al. (2024a) NN baseline is the first milestone:
trained NN should reach RMS ~0.17 and near-zero bias on the test set.

Run from code/:
    python -m scripts.train --model nn
    python -m scripts.train --model mc_dropout --dropout 0.1
 
Save the trained model, the standardizer and the training history under
results/<run_name>/ so evaluation and the thesis can reproduce every number.
"""
 
from __future__ import annotations
 
import argparse
import json
import time

import sys
from pathlib import Path
 
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
 
from src.data import FEATURES, ROOT, Standardizer, load_all, to_arrays
from src.models import build_model, nn_loss
 
RESULTS_DIR = ROOT / "code" / "results"
SEED = 42
 
 
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
def get_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
 
 
# =============================================================================
#  Data loading
# =============================================================================
 
def make_loaders(batch_size: int = 256):
    """Load splits, standardize on train, return loaders + arrays + scaler."""
    splits = load_all(verbose=True)
 
    X_tr, y_tr = to_arrays(splits["train"], FEATURES)
    X_va, y_va = to_arrays(splits["val"], FEATURES)
 
    scaler = Standardizer().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_va = scaler.transform(X_va)
 
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
 
    return train_loader, val_loader, scaler, splits
 
 
# =============================================================================
#  Train / validate one epoch
# =============================================================================
 
def loss_fn(name: str):
    """Return the loss function for a model."""
    # Extended for MDN/BNN later.
    if name in ("nn", "mc_dropout"):
        return nn_loss
    raise ValueError(f"No loss defined for model {name!r}")
 
 
def run_epoch(model, loader, device, criterion, optimizer=None) -> float:
    """One pass. Trains if optimizer is given, else evaluates. Returns mean loss."""
    is_train = optimizer is not None
    model.train(is_train)
    total, n = 0.0, 0
 
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n
 
 
# =============================================================================
#  Main training loop
# =============================================================================
 
def train(
    model_name: str = "nn",
    lr: float = 5e-4,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 256,
    dropout: float = 0.0,
    seed: int = SEED,
) -> Path:
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")
 
    train_loader, val_loader, scaler, _ = make_loaders(batch_size)
 
    model = build_model(model_name, in_dim=len(FEATURES), dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}  ({n_params:,} parameters)")
 
    criterion = loss_fn(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    run_name = f"{model_name}_seed{seed}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
 
    best_val = float("inf")
    best_epoch = -1
    history: list[dict] = []
    t0 = time.time()
 
    for epoch in range(1, max_epochs + 1):
        tr_loss = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss = run_epoch(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "train": tr_loss, "val": va_loss})
 
        improved = va_loss < best_val - 1e-5
        if improved:
            best_val, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), out_dir / "model.pt")
 
        if epoch == 1 or epoch % 10 == 0 or improved:
            flag = "  *" if improved else ""
            print(f"  epoch {epoch:3d}  train {tr_loss:.5f}  val {va_loss:.5f}{flag}")
 
        if epoch - best_epoch >= patience:
            print(f"  early stop at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break
 
    elapsed = time.time() - t0
    print(f"Best val loss {best_val:.5f} at epoch {best_epoch}  "
          f"({elapsed:.1f}s)")
 
    # save standardizer and metadata alongside the model
    scaler.save(out_dir / "scaler.npz")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "model": model_name, "lr": lr, "max_epochs": max_epochs,
        "patience": patience, "batch_size": batch_size, "dropout": dropout,
        "seed": seed, "features": FEATURES, "n_params": n_params,
        "best_epoch": best_epoch, "best_val": best_val,
        "device": str(device), "elapsed_sec": elapsed,
    }, indent=2))
 
    print(f"Saved to {out_dir}")
    return out_dir
 
 
# =============================================================================
#  CLI
# =============================================================================
 
def main() -> None:
    p = argparse.ArgumentParser(description="Train a photo-z model.")
    p.add_argument("--model", default="nn", choices=["nn", "mc_dropout"])
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
 
    train(
        model_name=args.model,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        dropout=args.dropout,
        seed=args.seed,
    )
 
if __name__ == "__main__":
    main()