"""Train a dense output decoder for coarse DiT tokens and compare held-out errors."""
import argparse
import random

import torch

from .evaluate import load_student
from .lora import add_lora
from .runtime import GPUBudget, local_path, write_json
from .train import load_cache, loss_terms, random_layout
from .upstream import ROOT


@torch.no_grad()
def validate(model, data, limit=96, count=8):
    model.eval()
    indices = torch.linspace(0, len(data[0]) - 1, min(limit, len(data[0]))).long()
    prior_enabled = model.reconstruction_enabled
    scores = {"repeat": [], "decoded": [], "decoded_coarse_region": [], "repeat_coarse_region": []}
    for batch_index, ids in enumerate(indices.split(4)):
        x, t, y, target = [value[ids].cuda() for value in data]
        layout = random_layout(len(ids), count, random.Random(6220 + batch_index), False, "cuda")
        active = layout.coarse.repeat_interleave(4, 1).repeat_interleave(4, 2)
        active_latents = active.repeat_interleave(2, 1).repeat_interleave(2, 2)[:, None]
        model.reconstruction_enabled = False
        repeated = model(x, t, y, layout)
        model.reconstruction_enabled = True
        decoded = model(x, t, y, layout)
        scores["repeat"].append((repeated[:, :4] - target[:, :4]).square().mean().item())
        scores["decoded"].append((decoded[:, :4] - target[:, :4]).square().mean().item())
        denom = active_latents.expand(-1, 4, -1, -1).sum().clamp_min(1)
        scores["repeat_coarse_region"].append(
            ((repeated[:, :4] - target[:, :4]).square() * active_latents).sum().item() / denom.item()
        )
        scores["decoded_coarse_region"].append(
            ((decoded[:, :4] - target[:, :4]).square() * active_latents).sum().item() / denom.item()
        )
    model.reconstruction_enabled = prior_enabled
    return {key: sum(values) / len(values) for key, values in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/training/detail/best.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--run-name", default="dense_output")
    parser.add_argument("--cache-dir", default="artifacts/cache")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.steps <= 0 or args.hours <= 0 or args.count not in (4, 8, 12, 16):
        raise ValueError("Steps and hours must be positive")

    source = local_path(args.checkpoint)
    if not source.exists():
        raise FileNotFoundError(source)
    directory = local_path(f"artifacts/training/{args.run_name}")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "best.pt").exists():
        raise FileExistsError("Use a new run name; existing reconstruction results are immutable")

    torch.manual_seed(6220)
    rng = random.Random(6220)
    source_state = torch.load(source, map_location="cpu", weights_only=True)
    if source_state.get("lora_rank", 0):
        raise ValueError("A reconstruction run must start from a checkpoint without LoRA")
    model = load_student("trained", source)
    if args.lora_rank:
        if args.lora_rank < 1:
            raise ValueError("LoRA rank must be positive")
        add_lora(model, args.lora_rank)
    model.reconstruction_enabled = True
    model.output_decoder.requires_grad_(True)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    train = load_cache("train", args.cache_dir)
    validation = load_cache("validation", args.cache_dir)
    config = vars(args) | dict(
        learning_rate=1e-4, effective_batch=16, microbatch=4, precision="fp32",
        train_records=len(train[0]), validation_records=len(validation[0]),
        trainable_parameters=sum(parameter.numel() for parameter in params),
        lora_rank=args.lora_rank,
        objective=f"teacher output residuals on {args.count} selected coarse blocks",
    )
    write_json(directory / "config.json", config)

    with GPUBudget("dense_output_training", args.hours) as budget:
        model.cuda()
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.)
        baseline = validate(model, validation, count=args.count)
        best = baseline["decoded"]
        log = [dict(step=0, validation=baseline)]
        torch.save(dict(model=model.state_dict(), lora_rank=args.lora_rank, step=0,
                        reconstruction_enabled=True, validation=baseline), directory / "best.pt")
        for step in range(1, args.steps + 1):
            budget.check()
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for _ in range(4):
                ids = torch.tensor([rng.randrange(len(train[0])) for _ in range(4)])
                x, t, y, target = [value[ids].cuda() for value in train]
                layout = random_layout(4, args.count, rng, step % 2 == 0, "cuda")
                prediction = model(x, t, y, layout)
                loss = loss_terms(prediction, target) / 4
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite reconstruction loss")
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 25 == 0:
                print(f"step={step} loss={total:.6f}", flush=True)
            if step % 100 == 0 or step == args.steps:
                scores = validate(model, validation, count=args.count)
                log.append(dict(step=step, loss=total, validation=scores))
                if scores["decoded"] < best:
                    best = scores["decoded"]
                    torch.save(dict(model=model.state_dict(), lora_rank=args.lora_rank, step=step,
                                    reconstruction_enabled=True, validation=scores), directory / "best.pt")
                torch.save(dict(model=model.state_dict(), lora_rank=args.lora_rank, step=step,
                                reconstruction_enabled=True, validation=scores), directory / "last.pt")
                write_json(directory / "validation.json", log)
                print(f"validation step={step}: {scores}", flush=True)
        write_json(directory / "validation.json", log)
        print(f"Saved {directory / 'best.pt'}; trainable parameters={sum(p.numel() for p in params)}", flush=True)


if __name__ == "__main__":
    main()
