import inspect
from pathlib import Path
import torch
from apt_dit import mixed_train as mt
from apt_dit.upstream import imports

RUN = Path('artifacts/xl512_lora32_10k')
SHARDS = RUN / 'teacher_shards'
SHARDS.mkdir(parents=True, exist_ok=True)
imports()
from models import DiT_models

model = DiT_models['DiT-XL/2'](input_size=64, num_classes=1000)
weights = torch.load('artifacts/checkpoints/DiT-XL-2-512x512.pt', map_location='cpu', weights_only=True)
model.load_state_dict(weights.get('ema', weights), strict=True)
del weights
model.eval().requires_grad_(False).cuda()

def teacher(*args):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(*args)

source = inspect.getsource(mt.generate)
changes = {
    'torch.randn(4, 32, 32,': 'torch.randn(4, 64, 64,',
    'rng = random.Random(6220)': 'rng = random.Random(6220)\n    for _ in range(seed_offset - 1_000_000):\n        rng.randrange(1000)',
    'traj=torch.cat(trajs)[:n_traj]': 'traj=torch.cat(trajs)[:n_traj] if n_traj else torch.empty((0, 51, 4, 64, 64), dtype=torch.float16)',
    'torch.save(data, path)': 'assert all(torch.isfinite(data[key]).all() for key in ("x0", "traj"))\n    tmp = path.with_suffix(".partial")\n    torch.save(data, tmp)\n    tmp.replace(path)',
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(source, mt.__dict__)
print('Teacher loaded. Target: 10,000 samples and 1,000 trajectories.', flush=True)
for first in range(0, 10_000, 250):
    mt.DATA = SHARDS / f'{first:05d}'
    n_traj = 250 if first < 1_000 else 0
    data = mt.generate(teacher, 250, n_traj, batch=10, seed_offset=1_000_000 + first)
    assert data['x0'].shape == (250, 4, 64, 64)
    # Archived shards can include extra trajectories beyond the new budget.
    assert data['traj'].shape[1:] == (51, 4, 64, 64)
    assert len(data['traj']) >= n_traj
    assert data['seeds'].tolist() == list(range(1_000_000 + first, 1_000_250 + first))
    print(f'Committed shard {first // 250 + 1}/40; {first + 250}/10000 samples', flush=True)
    del data
(RUN / 'teacher.complete').touch()
print('Teacher generation complete.', flush=True)
