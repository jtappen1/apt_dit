import ast
from datetime import datetime
import math
from pathlib import Path
import re
import subprocess
import time

run = Path('runs/xl512_lora32_10k')
raw = run / 'train.log'
output = run / 'progress.log'
update_pattern = re.compile(r'^Update (\d+)/(\d+) loss=(\S+)')


def render(lines, active):
    out = ['DiT-XL/2 512 | LoRA-32 | 10,000 teacher samples | effective batch 256']
    latest = None
    step = None
    started = None
    for line in lines:
        if 'Teacher cache ready. Starting or resuming LoRA training.' in line:
            if started is None:
                started = datetime.fromisoformat(line.split()[0])
            out.append(line)
        elif line.startswith('Starting XL/512'):
            step = int(re.search(r'at update (\d+)', line).group(1))
            out.append(line)
        elif match := update_pattern.match(line):
            step, target = map(int, match.groups()[:2])
            loss = match.group(3)
            lr = 3e-4 * (1 + math.cos(math.pi * (step - 1) / target)) / 2
            latest = f'step={step}/{target} loss={loss} lr={lr:.2e}'
            if step == 1 or step % 100 == 0 or step == target:
                out.append(latest)
        elif line.startswith('Held-out loss: '):
            losses = ast.literal_eval(line.split(': ', 1)[1])
            parts = ' '.join(f'uniform{size}={loss:.5f}' for size, loss in losses.items())
            out.append(f'val step={step}: {parts}')
        elif line.startswith(('Validation update ', 'Early stopped ', 'Traceback', 'RuntimeError:', 'torch.OutOfMemoryError:')):
            out.append(line)
    if latest:
        elapsed = ''
        if started is not None:
            end = datetime.now(started.tzinfo) if active else datetime.fromtimestamp(raw.stat().st_mtime, started.tzinfo)
            elapsed = f' | elapsed wall time: {int((end - started).total_seconds())}s'
        out += ['', f'Latest {latest}{elapsed}']
    if (run / 'training.complete').exists():
        out.append('Status: training complete. See best.pt for the lowest held-out loss.')
    elif not active:
        out.append('Status: trainer process exited. See train.log for details.')
    else:
        out.append('Status: training running')
    return '\n'.join(out) + '\n'


while True:
    active = subprocess.run(['pgrep', '-f', r'[/]python -u (runs|scripts)/xl512_lora32_10k/train[.]py'], stdout=subprocess.DEVNULL).returncode == 0
    summary = render(raw.read_text().splitlines(), active)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(summary)
    temporary.replace(output)
    if not active or (run / 'training.complete').exists():
        break
    time.sleep(20)
