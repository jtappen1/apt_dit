import pytest
import torch
from torch.nn import functional as F

from apt_dit.patching import AdaptiveDiT, Layout, packing_indices, uniform_layout
from apt_dit.schedules import block_variation, schedule_counts, select_layout
from apt_dit.lora import add_lora, merge_lora
from apt_dit.upstream import imports


@pytest.fixture
def model():
    DiT, _ = imports()
    torch.manual_seed(1)
    base = DiT(input_size=32, patch_size=2, hidden_size=32, depth=2, num_heads=4)
    # Upstream initializes the output and adaLN to zero: randomize them so
    # parity tests cannot pass merely because both paths return zero tensors.
    for block in base.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=.02)
    torch.nn.init.normal_(base.final_layer.linear.weight, std=.02)
    return AdaptiveDiT(base.eval()).eval()


def inputs():
    return torch.randn(2, 4, 32, 32), torch.tensor([500, 200]), torch.tensor([1, 1000])


def test_all_fine_matches_original(model):
    x, t, y = inputs()
    actual = model(x, t, y, uniform_layout(2, False, "cpu"))
    torch.testing.assert_close(actual, model.base(x, t, y), rtol=0, atol=0)
    assert actual.abs().max() > 0


@pytest.mark.parametrize("count", [0, 1, 4, 8, 12, 16])
def test_spatial_restore_is_exact(count):
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, :count] = True
    mask[1, 16-count:] = True
    layout = Layout(mask.reshape(2, 4, 4), count)
    fi, ci, restore = packing_indices(layout)
    assert fi.shape[1] + ci.shape[1] == layout.tokens
    for batch in range(2):
        for r in range(16):
            for c in range(16):
                packed = restore[batch, r * 16 + c].item()
                if mask.reshape(2, 4, 4)[batch, r//4, c//4]:
                    assert ci[batch, packed-fi.shape[1]] == (r//2)*8+c//2
                else:
                    assert fi[batch, packed] == r*16+c


def test_coarse_matches_manual_resize_repeat(model):
    x, t, y = inputs()
    b = model.base
    reduced = F.interpolate(x, (16, 16), mode="bilinear", align_corners=False)
    z = b.x_embedder.proj(reduced).flatten(2).transpose(1, 2) + model.coarse_pos
    condition = b.t_embedder(t) + b.y_embedder(y, False)
    for block in b.blocks:
        z = block(z, condition)
    z = z.reshape(2, 8, 8, -1).repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(2, 256, -1)
    expected = b.unpatchify(b.final_layer(z, condition))
    torch.testing.assert_close(model(x, t, y, uniform_layout(2, True, "cpu")), expected)


def test_zero_branch_and_training_gradients(model):
    x, t, y = inputs()
    layout = select_layout(x, 8)
    before = model(x, t, y, layout)
    model.enable_detail_training()
    torch.testing.assert_close(before, model(x, t, y, layout), rtol=0, atol=0)
    model(x, t, y, layout).square().mean().backward()
    assert model.detail.residual.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    # Zero residual initially blocks gradients to the aggregation convolution.
    assert model.detail.aggregate.weight.grad.abs().sum() == 0


def test_lora_merge_preserves_predictions(model):
    x, t, y = inputs()
    layout = select_layout(x, 8)
    before = model(x, t, y, layout)
    add_lora(model, rank=4)
    torch.testing.assert_close(before, model(x, t, y, layout))
    for name, p in model.named_parameters():
        if name.endswith(".b"):
            torch.nn.init.normal_(p, std=.01)
    expected = model(x, t, y, layout)
    merge_lora(model)
    torch.testing.assert_close(expected, model(x, t, y, layout))


def test_selector_preserves_detailed_region():
    clean = torch.zeros(1, 4, 32, 32)
    clean[:, :, :8, :8] = torch.randn(1, 4, 8, 8)
    layout = select_layout(clean, 15)
    assert not layout.coarse[0, 0, 0]
    assert layout.coarse.sum() == 15
    a = select_layout(clean, 8, random=True, generator=torch.Generator().manual_seed(42))
    b = select_layout(clean, 8, random=True, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a.coarse, b.coarse)


def test_validation_and_warmup():
    with pytest.raises(ValueError):
        Layout(torch.zeros(1, 4, 4, dtype=torch.bool), 8)
    for arm in ("spatial", "random", "temporal", "joint"):
        counts = schedule_counts(arm)
        assert len(counts) == 50 and counts[:4] == [0]*4
    assert len(set(schedule_counts("joint")[4:])) == 2


def test_repeat_head_lower_bound(model):
    from apt_dit.diagnostics import repeat_projection
    x, t, y = inputs()
    layout = select_layout(x, 8)
    target = torch.randn(2, 8, 32, 32)
    projection = repeat_projection(target, layout)
    actual = model(x, t, y, layout)
    # Every model output belongs to the repeated-output subspace.
    torch.testing.assert_close(repeat_projection(actual, layout), actual)
    torch.testing.assert_close(repeat_projection(projection, layout), projection)
    assert (actual-target).square().mean() >= (projection-target).square().mean()
