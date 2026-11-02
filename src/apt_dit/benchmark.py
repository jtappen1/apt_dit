"""Repeated, synchronized single-image latency for complete sampling paths."""
import argparse
import statistics
import torch

from .evaluate import CHECKPOINT, load_student, load_vae
from .runtime import GPUBudget, write_json
from .sampling import sample
from .upstream import load_dit


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/training/detail/best.pt")
    args = p.parse_args()
    vae = load_vae()
    results = []
    with GPUBudget("latency_benchmark", .15) as budget:
        vae.cuda()
        for stage in ("baseline", "zero_shot", "trained"):
            model = load_dit(CHECKPOINT) if stage == "baseline" else load_student(stage, args.checkpoint)
            model.cuda()
            for arm in (["baseline"] if stage == "baseline" else ["fine", "coarse", "spatial", "random", "temporal", "joint"]):
                measurements = []
                for repeat in range(13):
                    latent, metrics, _ = sample(model, seed=999, class_id=1, arm=arm, budget=budget)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    vae.decode(latent/.18215)
                    end.record()
                    torch.cuda.synchronize()
                    if repeat >= 3:
                        measurements.append(dict(denoise=metrics["denoise_seconds"], decode=start.elapsed_time(end)/1000))
                timings = [m["denoise"] for m in measurements]
                row = dict(stage=stage, arm=arm, repeats=10, warmup=3,
                           median_seconds=statistics.median(timings), min_seconds=min(timings), max_seconds=max(timings),
                           median_end_to_end_seconds=statistics.median(m["denoise"]+m["decode"] for m in measurements),
                           raw=measurements)
                results.append(row)
                print(f"{stage}/{arm}: {row['median_seconds']:.4f}s", flush=True)
            del model
        write_json("results/latency.json", results)


if __name__ == "__main__":
    main()
