"""Distill a patch-4 DiT-S path from cached frozen-teacher predictions."""
import argparse
import random

import torch

from .ddit import DDiT
from .runtime import GPUBudget, local_path, write_json
from .upstream import ROOT, load_dit
from .evaluate import CHECKPOINT


def load_split(split):
    root = local_path(f"artifacts/cache/{split}")
    records = []
    for path in sorted(root.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for record in payload["records"]:
            for branch in range(len(record["x"])):
                records.append(tuple(record[key][branch] for key in ("x", "t", "y", "target")))
    if not records:
        raise FileNotFoundError(f"No cached {split} teacher records at {root}")
    return tuple(torch.stack([r[i] for r in records]) for i in range(4))


@torch.no_grad()
def validate(model, data, limit=128, batch_size=4):
    model.eval()
    ids = torch.linspace(0, len(data[0]) - 1, min(limit, len(data[0]))).round().long()
    losses = []
    for selected in ids.split(batch_size):
        x, t, y, target = [value[selected].cuda() for value in data]
        prediction = model(x, t, y, patch_size=4)
        losses.append((prediction - target).square().mean().item())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--hours", type=float, default=2.)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--run-name", default="ddit_s_p4")
    args = parser.parse_args()
    if min(args.steps, args.batch_size) <= 0 or args.hours <= 0:
        raise ValueError("Steps, batch size, and hours must be positive")
    directory = local_path(f"artifacts/training/{args.run_name}")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "best.pt").exists():
        raise FileExistsError(f"Use a new run name; {directory / 'best.pt'} already exists")

    torch.manual_seed(6220)
    rng = random.Random(6220)
    model = DDiT(load_dit(CHECKPOINT), rank=32).cuda().eval()
    train, validation_data = load_split("train"), load_split("validation")
    params = model.trainable_parameters()
    write_json(directory / "config.json", dict(
        model="DiT-S/2 plus patch-4 path", patch_sizes=[2, 4], lora_rank=32,
        steps=args.steps, batch_size=args.batch_size, learning_rate=1e-4,
        train_records=len(train[0]), validation_records=len(validation_data[0]),
        trainable_parameters=sum(p.numel() for p in params),
        objective="patch-4 teacher prediction MSE; frozen DiT-S teacher",
    ))
    with GPUBudget("ddit_patch4_distillation", args.hours) as budget:
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.)
        best = validate(model, validation_data)
        logs = [dict(step=0, validation_mse=best)]
        torch.save(dict(model=model.state_dict(), rank=32, step=0, validation_mse=best), directory / "best.pt")
        print(f"initial heldout teacher MSE={best:.7f}", flush=True)
        for step in range(1, args.steps + 1):
            budget.check()
            ids = torch.tensor([rng.randrange(len(train[0])) for _ in range(args.batch_size)])
            x, t, y, target = [value[ids].cuda() for value in train]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, t, y, patch_size=4)
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite DDiT loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 25 == 0:
                print(f"step={step} train_mse={loss.item():.7f}", flush=True)
            if step % 100 == 0 or step == args.steps:
                score = validate(model, validation_data)
                logs.append(dict(step=step, train_mse=loss.item(), validation_mse=score))
                if score < best:
                    best = score
                    torch.save(dict(model=model.state_dict(), rank=32, step=step, validation_mse=score), directory / "best.pt")
                torch.save(dict(model=model.state_dict(), rank=32, step=step, validation_mse=score), directory / "last.pt")
                write_json(directory / "validation.json", logs)
                print(f"validation step={step} teacher_mse={score:.7f} best={best:.7f}", flush=True)
        write_json(directory / "validation.json", logs)
        print(f"saved {directory / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
