import torch
import pytest

from apt_dit.upstream import ROOT, load_dit

CKPT = ROOT / "artifacts/checkpoints/dit_s_256.pt"
pytestmark = pytest.mark.skipif(not CKPT.exists() or not torch.cuda.is_available(), reason="needs GPU + checkpoint")


@pytest.fixture(scope="module")
def model():
    from apt_dit.mixed import MixedPatchDiT
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = MixedPatchDiT(load_dit(CKPT), rank=32).cuda().eval()
    for lo in m.lora:  # make LoRA nonzero so the coarse-only gating is tested
        for mod in lo.values():
            torch.nn.init.normal_(mod.b, std=1e-2)
    return m


def test_sincos_p2_matches_pretrained(model):
    from apt_dit.mixed import size_positions
    pos = size_positions(model.width, 2).cuda()
    # Checkpoint pos_embed was stored at lower precision; max diff ~1.6e-4.
    torch.testing.assert_close(pos, model.base.pos_embed[0].float(), rtol=0, atol=5e-4)


@torch.no_grad()
def test_all_fine_general_path_matches_teacher(model):
    x = torch.randn(3, 4, 32, 32, device="cuda")
    t = torch.tensor([10, 500, 990], device="cuda")
    y = torch.tensor([1, 2, 1000], device="cuda")
    lay = torch.full((3, 4, 4), 2, device="cuda")
    expected = model.base(x, t, y)
    # Force the general (non-shortcut) path: tokens + LoRA gating + scatter + heads.
    model.full_finetune = True
    try:
        got = model(x, t, y, lay)
    finally:
        model.full_finetune = False
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_padding_matches_individual(model):
    x = torch.randn(3, 4, 32, 32, device="cuda")
    t = torch.tensor([100, 400, 800], device="cuda")
    y = torch.tensor([5, 6, 7], device="cuda")
    lay = torch.tensor([2, 4, 8], device="cuda")[torch.randint(0, 3, (3, 4, 4), device="cuda")]
    lay[0] = 8
    batched = model(x, t, y, lay)
    single = torch.cat([model(x[i:i + 1], t[i:i + 1], y[i:i + 1], lay[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(batched, single, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_sampler_matches_upstream(model):
    from apt_dit.mixed_sample import sample_batch
    from apt_dit.sampling import sample
    lat, _ = sample_batch(model, [123], [207], arm="fine", steps=50, base=model.base)
    ref, _, _ = sample(model.base, seed=123, class_id=207, arm="baseline", steps=50)
    torch.testing.assert_close(lat, ref, rtol=1e-3, atol=1e-3)
