"""Teacher-output distillation of the APT detail branch; LoRA is opt-in."""
import argparse
import random

import torch

from .evaluate import load_student
from .lora import add_lora
from .patching import Layout, uniform_layout
from .runtime import GPUBudget, local_path, write_json
from .upstream import ROOT


def load_cache(split, cache_root="artifacts/cache"):
    entries = []
    root = local_path(cache_root) / split
    for path in sorted(root.glob("*.pt")):
        cached = torch.load(path, map_location="cpu", weights_only=True)
        for record in cached["records"]:
            # Warmup too: all-coarse endpoint requires coverage at early steps.
            for branch in (0, 1):
                entries.append(tuple(record[key][branch] for key in ("x", "t", "y", "target")))
    if not entries:
        raise FileNotFoundError(f"No {split} teacher cache")
    return tuple(torch.stack([entry[i] for entry in entries]) for i in range(4))


def random_layout(batch, count, rng, contiguous, device):
    mask = torch.zeros(batch, 16, dtype=torch.bool)
    for i in range(batch):
        if contiguous:
            chosen = {rng.randrange(16)}
            while len(chosen) < count:
                neighbors = sorted({v for j in chosen for v in (j-4, j+4, j-1, j+1)
                                    if 0 <= v < 16 and abs(v//4-j//4)+abs(v%4-j%4) == 1} - chosen)
                chosen.add(rng.choice(neighbors))
            positions = sorted(chosen)
        else:
            positions = rng.sample(range(16), count)
        mask[i, positions] = True
    return Layout(mask.reshape(batch, 4, 4).to(device), count)


def loss_terms(prediction, target):
    return (prediction[:, :4]-target[:, :4]).square().mean() + (prediction[:, 4:]-target[:, 4:]).square().mean()


@torch.no_grad()
def validate(model, data, limit=64):
    model.eval()
    # Fixed stratified positions across the entire cache, not just first classes.
    indices = torch.linspace(0, len(data[0])-1, min(limit, len(data[0]))).long()
    scores = {}
    for count in (0, 4, 8, 12, 16):
        rng, losses = random.Random(100+count), []
        for ids in indices.split(4):
            x, t, y, target = [v[ids].cuda() for v in data]
            layout = uniform_layout(len(ids), False, "cuda") if count == 0 else random_layout(len(ids), count, rng, False, "cuda")
            losses.append(loss_terms(model(x, t, y, layout), target).item())
        scores[str(count)] = sum(losses) / len(losses)
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--hours", type=float, default=3.)
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--resume")
    p.add_argument("--run-name", default="detail")
    args = p.parse_args()
    if args.steps <= 0 or args.hours <= 0:
        raise ValueError("Steps and hours must be positive")
    zero_report = ROOT / "results/zero_shot_report.md"
    snapshot = ROOT / "artifacts/zero_shot/initial.pt"
    if not zero_report.exists() or not snapshot.exists():
        raise RuntimeError("Finish and save zero-shot evaluation before any training")
    if args.lora_rank and not args.resume:
        raise ValueError("Optional LoRA must start from a completed detail-only checkpoint")
    torch.manual_seed(6220)
    rng = random.Random(6220)
    model = load_student()
    state = torch.load(local_path(args.resume) if args.resume else snapshot, map_location="cpu", weights_only=True)
    if state.get("lora_rank", 0):
        raise ValueError("This entry point resumes detail-only checkpoints; LoRA continuation is not implemented")
    model.load_state_dict(state["model"], strict=True)
    model.enable_detail_training()
    if args.lora_rank:
        add_lora(model, args.lora_rank)
    directory = local_path(f"artifacts/training/{args.run_name}")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "best.pt").exists():
        raise FileExistsError("Use a new run name; existing training results are immutable")
    train, validation = load_cache("train"), load_cache("validation")
    config = vars(args) | dict(learning_rate=1e-4, effective_batch=16, microbatch=4,
                               precision="fp32", training_records=len(train[0]), validation_records=len(validation[0]))
    write_json(directory / "config.json", config)
    with GPUBudget("training_"+args.run_name, args.hours) as budget:
        model.cuda()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.)
        baseline = validate(model, validation)
        best = sum(baseline.values()) / 5
        log = [dict(step=0, validation=baseline)]
        torch.save(dict(model=model.state_dict(), lora_rank=args.lora_rank, step=0, validation=baseline), directory / "best.pt")
        stale = 0
        for step in range(1, args.steps+1):
            budget.check()
            model.eval()  # Deterministic teacher conditioning; gradients remain enabled.
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            count = rng.choice((4, 8, 12, 16))
            for micro in range(4):
                ids = torch.tensor([rng.randrange(len(train[0])) for _ in range(4)])
                x, t, y, target = [v[ids].cuda() for v in train]
                layout = random_layout(4, count, rng, step % 2 == 0, "cuda")
                loss = loss_terms(model(x, t, y, layout), target) / 4
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 25 == 0:
                print(f"step={step} loss={total:.6f} coarse_blocks={count}", flush=True)
            if step % 100 == 0 or step == args.steps:
                scores = validate(model, validation)
                value = sum(scores.values()) / 5
                log.append(dict(step=step, loss=total, validation=scores))
                if value < best:
                    best, stale = value, 0
                    torch.save(dict(model=model.state_dict(), lora_rank=args.lora_rank, step=step, validation=scores), directory / "best.pt")
                else:
                    stale += 1
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), lora_rank=args.lora_rank,
                                step=step, validation=scores, rng_state=rng.getstate(), torch_rng=torch.get_rng_state()), directory / "last.pt")
                write_json(directory / "validation.json", log)
                print(f"validation step={step}: {scores}", flush=True)
                if stale >= 5:
                    print("Stopped after five validation checks without improvement", flush=True)
                    break
        write_json(directory / "validation.json", log)
        print(f"Saved {directory / 'best.pt'}; trainable parameters={sum(p.numel() for p in params)}", flush=True)


if __name__ == "__main__":
    main()
