"""Cache teacher predictions once, keeping all states from a trajectory together."""
import argparse
import torch

from .evaluate import CHECKPOINT
from .runtime import GPUBudget, manifest
from .sampling import sample
from .upstream import ROOT, load_dit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=128)
    p.add_argument("--validation", type=int, default=32)
    p.add_argument("--hours", type=float, default=.75)
    args = p.parse_args()
    with GPUBudget("teacher_cache", args.hours) as budget:
        model = load_dit(CHECKPOINT, "cuda")
        for split, count in (("train", args.train), ("validation", args.validation)):
            directory = ROOT / "artifacts/cache" / split
            directory.mkdir(parents=True, exist_ok=True)
            for entry in manifest()[split][:count]:
                path = directory / f"{entry['seed']}.pt"
                if path.exists():
                    continue
                budget.check()
                latent, metrics, records = sample(model, **entry, cache=True, budget=budget)
                torch.save(dict(entry=entry, latent=latent.cpu(), metrics=metrics, records=records), path)
                print(f"cached {split}/{entry['seed']}: {len(records)} states", flush=True)


if __name__ == "__main__":
    main()
