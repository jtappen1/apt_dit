"""Collect frozen-teacher targets on the current adaptive student's rollouts."""
import argparse
import json

import torch

from .evaluate import CHECKPOINT, load_student
from .runtime import GPUBudget, local_path, sha256, write_json
from .sampling import sample
from .upstream import ROOT, load_dit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="artifacts/training/dense_output/best.pt")
    parser.add_argument("--train", type=int, default=128)
    parser.add_argument("--validation", type=int, default=32)
    parser.add_argument("--hours", type=float, default=0.75)
    parser.add_argument("--cache-dir", default="artifacts/onpolicy_cache")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if min(args.train, args.validation) <= 0 or args.hours <= 0 or args.count not in (4, 8, 12, 16):
        raise ValueError("Cache sizes and hours must be positive")
    student_checkpoint = local_path(args.student)
    if not student_checkpoint.exists():
        raise FileNotFoundError(student_checkpoint)

    root = local_path(args.cache_dir)
    config = dict(student=str(student_checkpoint.relative_to(ROOT)),
                  student_sha256=sha256(student_checkpoint), teacher_sha256=sha256(CHECKPOINT),
                  arm="spatial", count=args.count, steps=50, cfg_scale=4., signal="clean",
                  train=args.train, validation=args.validation)
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cache directory already belongs to a different configuration")
    write_json(config_path, config)

    manifest_path = ROOT / "artifacts/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with GPUBudget("onpolicy_teacher_cache", args.hours) as budget:
        student = load_student("trained", student_checkpoint).cuda().eval()
        teacher = load_dit(CHECKPOINT, "cuda").eval()
        for split, count in (("train", args.train), ("validation", args.validation)):
            directory = root / split
            directory.mkdir(parents=True, exist_ok=True)
            for entry in manifest[split][:count]:
                budget.check()
                path = directory / f"{entry['seed']}.pt"
                if path.exists():
                    continue
                _, metrics, records = sample(student, **entry, arm="spatial", count=args.count,
                                             cache=True, budget=budget)
                for record in records:
                    budget.check()
                    x, t, y = (record[key].cuda() for key in ("x", "t", "y"))
                    with torch.no_grad():
                        record["target"] = teacher(x, t, y).cpu()
                torch.save(dict(entry=entry, metrics=metrics, records=records), path)
                print(f"cached on-policy {split}/{entry['seed']}: {len(records)} states", flush=True)
        print(f"Saved on-policy teacher targets under {root}", flush=True)


if __name__ == "__main__":
    main()
