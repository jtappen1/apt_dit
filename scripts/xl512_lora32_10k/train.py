import argparse
import inspect
import math
import sys
from pathlib import Path
from types import ModuleType

import torch
from apt_dit import mixed_train as mt
from apt_dit.upstream import imports

RUN = Path('runs/xl512_lora32_10k')
BASE = Path('artifacts/checkpoints/DiT-XL-2-512x512.pt')
GEOMETRY = Path(__file__).with_name('mixed_xl512.py')
ARTIFACTS = Path('artifacts/xl512_lora32_10k')
LAST = ARTIFACTS / 'latest.pt'


def load_model():
    imports()
    from models import DiT_models
    base = DiT_models['DiT-XL/2'](input_size=64, num_classes=1000)
    weights = torch.load(BASE, map_location='cpu', weights_only=True)
    base.load_state_dict(weights.get('ema', weights), strict=True)
    module = ModuleType('apt_dit.mixed_xl512_full')
    sys.modules[module.__name__] = module
    exec(compile(GEOMETRY.read_text(), str(GEOMETRY), 'exec'), module.__dict__)
    model = module.MixedPatchDiT(base, rank=32).cuda()
    model.train()
    base.eval()
    return base, model


def load_data():
    paths = sorted((Path('artifacts/xl512_lora32_10k') / 'teacher_shards').glob('*/*.pt'))
    assert (Path('artifacts/xl512_lora32_10k') / 'teacher.complete').exists() and len(paths) == 40
    data = {
        'x0': torch.empty(10_000, 4, 64, 64, device='cuda'),
        'classes': torch.empty(10_000, dtype=torch.long, device='cuda'),
        'traj': torch.empty(1_000, 51, 4, 64, 64, dtype=torch.float16, device='cuda'),
        'traj_classes': torch.empty(1_000, dtype=torch.long, device='cuda'),
    }
    for first, path in enumerate(paths):
        lo = first * 250
        shard = torch.load(path, map_location='cpu', weights_only=True)
        for key in data:
            if len(shard[key]) and (key not in ('traj', 'traj_classes') or lo < 1_000):
                data[key][lo:lo + 250].copy_(shard[key])
    return data


def make_batcher(data, teacher):
    source = inspect.getsource(mt.smooth_layout)
    for old, new in (('n, 1, 4, 4', 'n, 1, 8, 8'), ('/ 15', '/ 63'), ('n, 4, 4', 'n, 8, 8')):
        assert source.count(old) == 1
        source = source.replace(old, new)
    exec(source, mt.__dict__)
    return mt.Batcher(data, teacher)


def save(model, optimizer, batcher, step, early, path=LAST):
    adapter = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = dict(adapter=adapter, optimizer=optimizer.state_dict(), step=step, batch_rng=batcher.gen.get_state(),
                 torch_rng=torch.random.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                 base_checkpoint=str(BASE), rank=32, resolution=512, early=early)
    tmp = path.with_suffix('.partial')
    torch.save(state, tmp)
    tmp.replace(path)


@torch.no_grad()
def validate(model, teacher, batcher, states, micro):
    generator = torch.Generator(device='cuda').manual_seed(6412)
    frames = torch.randint(0, 50, (50,), device='cuda', generator=generator)
    rows = torch.arange(50, device='cuda')
    x = states[0][rows, frames].float()
    y = states[1]
    t = batcher.tmap.cuda()[49 - frames]
    losses = {4: 0., 8: 0.}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, 50, micro):
            sl = slice(start, start + micro)
            target = teacher(x[sl], t[sl], y[sl])
            for p in losses:
                layout = torch.full((len(x[sl]), 8, 8), p, device='cuda')
                pred = model(x[sl], t[sl], y[sl], layout).float()
                losses[p] += mt.loss_fn(pred, target).item() * len(x[sl]) / 50
    print(f'Held-out loss: {losses}', flush=True)
    return sum(losses.values()) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2517)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min-delta', type=float, default=0.001)
    args = parser.parse_args()
    assert args.microbatch >= 2 and args.microbatch % 2 == 0 and 256 % args.microbatch == 0
    torch.manual_seed(6220)
    torch.backends.cuda.matmul.allow_tf32 = True
    base, model = load_model()

    def teacher(x, t, y):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return base(x, t, y).float()

    print('Loading 10,000 samples and 1,000 native trajectories.', flush=True)
    data = load_data()
    heldout = data['traj'][-50:], data['traj_classes'][-50:]
    data['traj'], data['traj_classes'] = data['traj'][:-50], data['traj_classes'][:-50]
    batcher = make_batcher(data, teacher)
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=3e-4)
    if args.patience < 1 or args.min_delta < 0:
        parser.error('patience must be positive and min-delta must be nonnegative')
    start = 0
    early = dict(best=float('inf'), anchor=float('inf'), stale=0, stopped=False)
    if LAST.exists():
        state = torch.load(LAST, map_location='cpu', weights_only=True)
        actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert actual == set(state['adapter'])
        model.load_state_dict(state['adapter'], strict=False)
        optimizer.load_state_dict(state['optimizer'])
        batcher.gen.set_state(state['batch_rng'])
        torch.random.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        start = state['step']
        early.update(state.get('early', {}))
    stop = min(args.steps, args.stop_after or args.steps)
    accum = 256 // args.microbatch
    print(f'Starting XL/512 LoRA-32 at update {start}; target {args.steps}; effective batch 256.', flush=True)
    def check_validation(step):
        score = validate(model, teacher, batcher, heldout, args.microbatch)
        if not math.isfinite(score):
            raise RuntimeError(f'Nonfinite held-out loss at update {step}')
        improved = score < early['best']
        if improved:
            early['best'] = score
        if score < early['anchor'] - args.min_delta:
            early['anchor'] = score
            early['stale'] = 0
        else:
            early['stale'] += 1
        early['stopped'] = early['stale'] >= args.patience
        if improved:
            save(model, optimizer, batcher, step, early, ARTIFACTS / 'best.pt')
        save(model, optimizer, batcher, step, early)
        print(f'Validation update {step}: mean={score:.6f}, best={early["best"]:.6f}, '
              f'patience={early["stale"]}/{args.patience}', flush=True)
        return early['stopped']

    if early['stopped']:
        print('Early stopping was already reached. Best weights are in best.pt.', flush=True)
        return
    if start and not math.isfinite(early['best']):
        check_validation(start)
    for step in range(start, stop):
        optimizer.param_groups[0]['lr'] = 3e-4 * (1 + math.cos(math.pi * step / args.steps)) / 2
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device='cuda')
        for _ in range(accum):
            x, t, y, layout, target = batcher.get(args.microbatch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = mt.loss_fn(model(x, t, y, layout).float(), target)
            (loss / accum).backward()
            total += loss.detach() / accum
        assert torch.isfinite(total)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        print(f'Update {step + 1}/{args.steps} loss={total.item():.6f} grad_norm={norm.item():.5f}', flush=True)
        update = step + 1
        if update % 250 == 0 or update in (1, stop):
            if check_validation(update):
                print(f'Early stopped at update {update}. Best weights are in best.pt.', flush=True)
                (RUN / 'training.complete').touch()
                return
        elif update % 25 == 0:
            save(model, optimizer, batcher, update, early)
    if stop == args.steps:
        (RUN / 'training.complete').touch()


if __name__ == '__main__':
    main()
