"""Stage 0: compute headroom of token reduction at 256 / 1024 / 2048 px (no training).

Times the real transformer trunk (all blocks: self-attn, cross-attn, MLP, adaLN) at
full and reduced token counts, with CFG (2 sequences per image), and counts executed
FLOPs with torch.utils.flop_counter, splitting out attention. This is an upper bound
on what any token-reducing schedule can save; it excludes embed/head/selector costs.

  DiT-S/2 @256      (reference: our main study)        N = 256
  PixArt-alpha 1024 (proposed Stage 1-2 target)        N = 4096
  PixArt-Sigma 2K                                       N = 16384

Pre-registered gate: PixArt-alpha 1024, bf16, one image (CFG batch 2):
trunk speedup at 50% tokens >= 1.5x  -> build Stages 1-2.
"""
import json
import statistics

import torch
from torch.utils.flop_counter import FlopCounterMode

from .upstream import ROOT, load_dit

FRACS = [1.0, 0.75, 0.5, 0.25, 1 / 16]
MODELS = {
    "pixart_alpha_1024": dict(repo="PixArt-alpha/PixArt-XL-2-1024-MS", latent=128, text_len=120, extra=True),
    "pixart_sigma_2048": dict(repo="PixArt-alpha/PixArt-Sigma-XL-2-2K-MS", latent=256, text_len=300, extra=False),
}


def timed(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / 1000)
    return statistics.median(ts)


def flops_split(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    total = fc.get_total_flops()
    attn = 0
    for op, n in fc.get_flop_counts()["Global"].items():
        name = str(op)
        if "scaled_dot_product" in name or "bmm" in name:
            attn += n
    return total, attn


@torch.no_grad()
def pixart_rows(key, spec, batches):
    from diffusers import PixArtTransformer2DModel
    m = PixArtTransformer2DModel.from_pretrained(spec["repo"], subfolder="transformer",
                                                 torch_dtype=torch.bfloat16).cuda().eval()
    d = m.config.num_attention_heads * m.config.attention_head_dim
    N = (spec["latent"] // m.config.patch_size) ** 2
    rows = []
    for imgs in batches:
        B = 2 * imgs  # CFG: conditional + unconditional sequences
        t = torch.full((B,), 500, device="cuda")
        added = None
        if spec["extra"]:
            px = spec["latent"] * 8
            added = dict(resolution=torch.tensor([[px, px]] * B, device="cuda", dtype=torch.bfloat16),
                         aspect_ratio=torch.ones(B, 1, device="cuda", dtype=torch.bfloat16))
        temb, _ = m.adaln_single(t, added, batch_size=B, hidden_dtype=torch.bfloat16)
        enc = m.caption_projection(torch.randn(B, spec["text_len"], 4096, device="cuda", dtype=torch.bfloat16))
        for f in FRACS:
            n = max(1, round(N * f))
            h0 = torch.randn(B, n, d, device="cuda", dtype=torch.bfloat16)

            def trunk(blocks=m.transformer_blocks):
                h = h0
                for blk in blocks:
                    h = blk(h, encoder_hidden_states=enc, timestep=temb)
                return h

            sec = timed(trunk)
            tot, att = flops_split(lambda: trunk(m.transformer_blocks[:1]))
            rows.append(dict(model=key, images=imgs, tokens=n, frac=f, seconds=sec,
                             tflops=tot * len(m.transformer_blocks) / 1e12, attn_share=att / tot))
            print(rows[-1], flush=True)
    del m
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def dits_rows(batches):
    base = load_dit(ROOT / "artifacts/checkpoints/dit_s_256.pt", "cuda")
    rows = []
    for imgs in batches:
        B = 2 * imgs
        c = base.t_embedder(torch.full((B,), 500, device="cuda")) + base.y_embedder(torch.randint(0, 1000, (B,), device="cuda"), False)
        for f in FRACS:
            n = max(1, round(256 * f))
            h0 = torch.randn(B, n, 384, device="cuda")

            def trunk(blocks=base.blocks):
                h = h0
                for blk in blocks:
                    h = blk(h, c)
                return h

            with torch.autocast("cuda", dtype=torch.bfloat16):
                sec = timed(trunk)
                tot, att = flops_split(lambda: trunk(base.blocks[:1]))
            rows.append(dict(model="dit_s_256", images=imgs, tokens=n, frac=f, seconds=sec,
                             tflops=tot * len(base.blocks) / 1e12, attn_share=att / tot))
            print(rows[-1], flush=True)
    return rows


def main():
    batches = [1, 4]
    rows = dits_rows(batches)
    for key, spec in MODELS.items():
        rows += pixart_rows(key, spec, batches)
    out = ROOT / "results/mixed"
    out.mkdir(parents=True, exist_ok=True)
    full = {(r["model"], r["images"]): r["seconds"] for r in rows if r["frac"] == 1.0}
    for r in rows:
        r["speedup"] = full[(r["model"], r["images"])] / r["seconds"]
    gate_row = next(r for r in rows if r["model"] == "pixart_alpha_1024" and r["images"] == 1 and r["frac"] == 0.5)
    gate = dict(rule="PixArt-alpha 1024, bf16, 1 image (CFG batch 2): trunk speedup at 50% tokens >= 1.5x",
                speedup=gate_row["speedup"], passed=gate_row["speedup"] >= 1.5)
    (out / "stage0_headroom.json").write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows, gate=gate), indent=2))
    lines = ["# Stage 0: compute headroom of token reduction (trunk only, bf16, CFG)", "", f"GPU: {torch.cuda.get_device_name()}", "",
             "Upper bound for any token-reducing schedule: real transformer blocks at reduced token counts; "
             "excludes embedding/head/selector/decoder overhead.", "",
             "| Model | Images | Tokens (frac) | Trunk TFLOPs | Attention share | Trunk time | Speedup vs full |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['images']} | {r['tokens']} ({r['frac']:.3g}) | {r['tflops']:.3f} | "
                     f"{r['attn_share']:.0%} | {r['seconds'] * 1e3:.2f} ms | {r['speedup']:.2f}× |")
    lines += ["", f"**Gate** ({gate['rule']}): speedup {gate['speedup']:.2f}× → "
              f"{'PASSED: build Stages 1–2' if gate['passed'] else 'FAILED: do not build Stages 1–2'}."]
    (out / "stage0_headroom.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
