"""Load the pinned official implementation without modifying its sources."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DIT_REVISION = "ed81ce2229091fd4ecc9a223645f95cf379d582b"
APT_REVISION = "707337ee66c7ff62b820538b40b75341d80e32ef"


def imports():
    path = ROOT / "vendor" / "DiT"
    if not (path / "models.py").exists():
        raise FileNotFoundError("Run scripts/setup_sources.sh first")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from models import DiT
    from diffusion import create_diffusion
    return DiT, create_diffusion


def load_dit(checkpoint, device="cpu"):
    import torch
    DiT, _ = imports()
    model = DiT(input_size=32, patch_size=2, hidden_size=384, depth=12, num_heads=6)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "ema" in weights:
        weights = weights["ema"]
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False).to(device)
