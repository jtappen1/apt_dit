"""Held-out prediction diagnostics, including a repeat-head error lower bound."""
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from .evaluate import load_student
from .patching import Layout, uniform_layout
from .runtime import GPUBudget, write_json
from .schedules import block_variation, select_layout, trajectory_variation
from .train import loss_terms
from .upstream import ROOT


def repeat_projection(target, layout):
    """Orthogonal projection onto the repeated-fine-patch output subspace.

    This uses teacher targets and is an unattainable-in-practice oracle floor,
    not an inference method. Fine areas are allowed to match targets exactly.
    """
    b, c, _, _ = target.shape
    tokens = target.reshape(b, c, 16, 2, 16, 2).permute(0, 2, 4, 3, 5, 1).reshape(b, 16, 16, 4*c)
    coarse = tokens.reshape(b, 8, 2, 8, 2, 4*c).mean((2, 4))
    repeated = coarse.repeat_interleave(2, 1).repeat_interleave(2, 2)
    mask = layout.coarse.repeat_interleave(4, 1).repeat_interleave(4, 2)[..., None]
    out = torch.where(mask, repeated, tokens)
    return out.reshape(b, 16, 16, 2, 2, c).permute(0, 5, 1, 3, 2, 4).reshape_as(target)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/training/detail/best.pt")
    args = p.parse_args()
    rows = []
    with GPUBudget("heldout_diagnostics", .3) as budget:
        for stage in ("zero_shot", "trained"):
            model = load_student(stage, args.checkpoint).cuda()
            for path in sorted((ROOT / "artifacts/cache/validation").glob("*.pt"))[:16]:
                cached = torch.load(path, weights_only=True)
                for record in [cached["records"][i] for i in (1, 4, 8)]:
                    budget.check()
                    x, t, y, target = [record[k].cuda() for k in ("x", "t", "y", "target")]
                    clean = record["clean"].cuda()
                    history = list(record["history"].cuda().split(1))
                    fine = model(x, t, y, uniform_layout(2, False, "cuda"))
                    fine_mse = (fine[:,:4]-target[:,:4]).square().mean().item()
                    damages = []
                    for block in range(16):
                        mask = torch.zeros(1, 16, dtype=torch.bool, device="cuda")
                        mask[0, block] = True
                        layout = Layout(mask.reshape(1, 4, 4), 1).cfg()
                        prediction = model(x, t, y, layout)
                        damages.append((prediction[:,:4]-target[:,:4]).square().mean().item() - fine_mse)
                    signals = dict(clean=block_variation(clean).flatten().cpu().numpy(),
                                   trajectory=trajectory_variation(history).flatten().cpu().numpy())
                    correlations = {}
                    for name, scores in signals.items():
                        rho = float(spearmanr(scores, damages).statistic)
                        correlations[name] = rho if np.isfinite(rho) else None
                    strategies = {name: select_layout(clean, 8, signal=name, history=history).cfg() for name in signals}
                    mask = torch.zeros(1, 16, dtype=torch.bool, device="cuda")
                    mask[0, np.argsort(damages)[:8].tolist()] = True
                    strategies["oracle_rank"] = Layout(mask.reshape(1, 4, 4), 8).cfg()
                    strategies["random"] = select_layout(clean, 8, random=True, generator=torch.Generator(device="cuda").manual_seed(cached["entry"]["seed"]+record["index"])).cfg()
                    results = {}
                    for name, layout in strategies.items():
                        prediction = model(x, t, y, layout)
                        projected = repeat_projection(target, layout)
                        results[name] = dict(loss=loss_terms(prediction, target).item(),
                                             noise_mse=(prediction[:,:4]-target[:,:4]).square().mean().item(),
                                             repeat_floor_noise_mse=(projected[:,:4]-target[:,:4]).square().mean().item())
                    rows.append(dict(stage=stage, seed=cached["entry"]["seed"], index=record["index"],
                                     timestep=int(t[0]), fine_noise_mse=fine_mse, correlations=correlations,
                                     single_block_damages=damages, strategies=results))
            del model
        write_json("results/diagnostics.json", rows)
        print(f"Saved {len(rows)} held-out diagnostic records", flush=True)


if __name__ == "__main__":
    main()
